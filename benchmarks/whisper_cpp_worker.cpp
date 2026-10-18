// Persistent public-C-API worker. Protocol details live in cpp_backend.py.
#include "whisper.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

static constexpr uint32_t max_samples = 16000u * 60u * 30u;
static std::string quote(const std::string &s) {
    std::ostringstream out; out << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) { const char *hex="0123456789abcdef"; out << "\\u00" << hex[c >> 4] << hex[c & 15]; }
        else out << c;
    }
    out << '"'; return out.str();
}
static bool read_bytes(void *dst, size_t n) { return bool(std::cin.read(static_cast<char *>(dst), n)); }
static uint32_t read_u32() {
    unsigned char b[4]; if (!read_bytes(b,4)) throw std::runtime_error("truncated request");
    return uint32_t(b[0]) | uint32_t(b[1])<<8 | uint32_t(b[2])<<16 | uint32_t(b[3])<<24;
}
static void reply(const std::string &s) {
    if (s.size() > 64u*1024u*1024u) throw std::runtime_error("response exceeds limit");
    uint32_t n=uint32_t(s.size()); unsigned char b[]={uint8_t(n),uint8_t(n>>8),uint8_t(n>>16),uint8_t(n>>24)};
    std::cout.write(reinterpret_cast<char *>(b),4); std::cout.write(s.data(),s.size()); std::cout.flush();
}
static void suppress_common_tokens(whisper_context *, whisper_state *, const whisper_token_data *, int,
                                   float *logits, void *user_data) {
    const auto &ids = *static_cast<const std::vector<int> *>(user_data);
    for (int id : ids) logits[id] = -std::numeric_limits<float>::infinity();
}
int main(int argc, char **argv) {
    if (argc != 9) { std::cerr << "model language threads beam condition timestamps task suppression_csv required\n"; return 2; }
    whisper_context *ctx=nullptr;
    try {
        auto cp=whisper_context_default_params(); cp.use_gpu=true; cp.flash_attn=true;
        ctx=whisper_init_from_file_with_params(argv[1],cp);
        if (!ctx) throw std::runtime_error("model initialization failed");
        const std::string lang=argv[2];
        if (whisper_lang_id(lang.c_str()) < 0) throw std::runtime_error("unknown explicit language");
        const int beam=std::stoi(argv[4]);
        std::vector<int> suppressed;
        std::stringstream csv(argv[8]); std::string item;
        while (std::getline(csv,item,',')) {
            size_t parsed=0; int id=std::stoi(item,&parsed);
            if (parsed != item.size() || id < 0 || id >= whisper_n_vocab(ctx)) throw std::runtime_error("invalid suppression token ID");
            suppressed.push_back(id);
        }
        auto p=whisper_full_default_params(beam > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
        p.n_threads=std::stoi(argv[3]); p.language=lang.c_str(); p.detect_language=false;
        // Public no_context clears history at request entry; n_max_text_ctx
        // controls conditioning between windows inside that request.
        p.no_context=true;
        if (std::string(argv[5])=="0") p.n_max_text_ctx=0;
        p.no_timestamps=std::string(argv[6])=="0";
        p.translate=std::string(argv[7])=="translate";
        p.print_progress=false; p.print_realtime=false; p.print_timestamps=false; p.print_special=false;
        p.token_timestamps=false; p.temperature=0; p.temperature_inc=0;
        p.entropy_thold=-1; p.logprob_thold=-1; p.no_speech_thold=.6f;
        p.suppress_blank=true; p.suppress_nst=false; p.greedy.best_of=1;
        p.logits_filter_callback=suppress_common_tokens;
        p.logits_filter_callback_user_data=&suppressed;
        p.beam_search.beam_size=beam; p.beam_search.patience=1; p.length_penalty=-1;
        p.vad=false;
        std::ostringstream ready;
        ready << "{\"ready\":true,\"suppression_count\":" << suppressed.size() << ",\"version\":" << quote(whisper_version())
          << ",\"system_info\":" << quote(whisper_print_system_info())
          << ",\"model_type\":" << quote(whisper_model_type_readable(ctx))
          << ",\"dimensions\":{\"n_vocab\":" << whisper_model_n_vocab(ctx)
          << ",\"n_audio_ctx\":" << whisper_model_n_audio_ctx(ctx)
          << ",\"n_audio_state\":" << whisper_model_n_audio_state(ctx)
          << ",\"n_audio_head\":" << whisper_model_n_audio_head(ctx)
          << ",\"n_audio_layer\":" << whisper_model_n_audio_layer(ctx)
          << ",\"n_text_ctx\":" << whisper_model_n_text_ctx(ctx)
          << ",\"n_text_state\":" << whisper_model_n_text_state(ctx)
          << ",\"n_text_head\":" << whisper_model_n_text_head(ctx)
          << ",\"n_text_layer\":" << whisper_model_n_text_layer(ctx) << "}}";
        reply(ready.str());
        std::unordered_map<uint32_t,std::vector<float>> clips; size_t total=0;
        while (true) {
            unsigned char op; if (!read_bytes(&op,1)) break;
            if (op==3) break;
            const uint32_t id=read_u32();
            if (op==1) {
                uint32_t n=read_u32();
                if (!n || n > max_samples || clips.count(id) || total+n>max_samples) throw std::runtime_error("invalid PCM upload size/id");
                std::vector<float> samples(n);
                if (!read_bytes(samples.data(),n*sizeof(float))) throw std::runtime_error("truncated PCM upload");
                for (float f:samples) if (!std::isfinite(f)) throw std::runtime_error("nonfinite PCM");
                total+=n; clips.emplace(id,std::move(samples)); reply("{\"uploaded\":true}"); continue;
            }
            if (op!=2 || !clips.count(id)) throw std::runtime_error("invalid request opcode/id");
            const auto &samples=clips.at(id);
            auto start=std::chrono::steady_clock::now();
            if (whisper_full(ctx,p,samples.data(),int(samples.size())) != 0) throw std::runtime_error("whisper_full failed");
            std::ostringstream segments,tokens; std::string text; bool first_token=true;
            segments << '['; tokens << '[';
            for (int i=0;i<whisper_full_n_segments(ctx);++i) {
                if (i) segments << ',';
                std::string segment_text=whisper_full_get_segment_text(ctx,i); text+=segment_text;
                segments << "{\"start\":" << whisper_full_get_segment_t0(ctx,i)*.01
                  << ",\"end\":" << whisper_full_get_segment_t1(ctx,i)*.01 << ",\"text\":" << quote(segment_text) << ",\"tokens\":[";
                for (int j=0;j<whisper_full_n_tokens(ctx,i);++j) {
                    if (j) segments << ','; if (!first_token) tokens << ','; first_token=false;
                    int token=whisper_full_get_token_id(ctx,i,j); segments << token; tokens << token;
                }
                segments << "]}";
            }
            segments << ']'; tokens << ']';
            double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            const char *detected=whisper_lang_str(whisper_full_lang_id(ctx));
            std::ostringstream result; result.precision(12);
            result << "{\"text\":" << quote(text) << ",\"language\":" << quote(detected ? detected : lang)
              << ",\"segments\":" << segments.str() << ",\"tokens\":" << tokens.str()
              << ",\"service_seconds\":" << elapsed << '}'; reply(result.str());
        }
        whisper_free(ctx); return 0;
    } catch (const std::exception &e) {
        if (ctx) whisper_free(ctx);
        try { reply("{\"error\":"+quote(e.what())+"}"); } catch (...) {}
        std::cerr << e.what() << '\n'; return 1;
    }
}
