# SaarthiX - AI-Powered Multilingual Video Generation Platform

## 🇮🇳 Breaking Language Barriers, One Video at a Time

SaarthiX is an AI-driven, multi-agent video generation platform designed specifically for India's diverse linguistic landscape. We transform text or raw data into mobile-first explainer videos that make critical information accessible to 1.4B+ Indians across 20+ official languages and hundreds of dialects.

---

## 🌟 Inspiration

India is home to incredible diversity - 1.4B+ people speaking 20+ official languages and hundreds of dialects. Yet critical information on healthcare, agriculture, and climate alerts often remains inaccessible or delayed. We were inspired to build a tool that breaks these barriers by creating AI-powered, multilingual videos in minutes, making knowledge fast, inclusive, and actionable.

---

## 🚀 What SaarthiX Does

SaarthiX is an AI-driven, multi-agent video generation platform that:

- 🎥 **Transforms text/data** into mobile-first explainer videos
- 🗣️ **Supports 20+ Indian languages** and regional dialects  
- 🤖 **Auto-generates** scripts, visuals, and voiceovers
- 📱 **Delivers ready-to-share** content for social media, TV, and education
- 🌐 **Optimized for low-bandwidth** and offline environments
- ♿ **Accessibility features** including sign language support

### Key Use Cases
- **Healthcare:** Medical awareness and preventive care guidance
- **Agriculture:** Farming techniques and weather advisories
- **Government:** Scheme information and public announcements
- **Education:** Skill development and literacy programs
- **Emergency:** Climate alerts and disaster preparedness

---

## 🏗️ Architecture

### Multi-Agent System Design

1. **Content Agent** 🔍
   - Validates and curates input data
   - Ensures fact-checking and cultural sensitivity
   - Handles source verification

2. **Script Agent** ✍️
   - Converts data into simple, local-language scripts
   - Adapts content for different literacy levels
   - Supports dialect-specific translations

3. **Video Agent** 🎬
   - Generates visuals, captions, and voiceovers
   - Creates mobile-optimized content
   - Handles multilingual subtitle generation

### Technical Stack

- **Languages:** Python, JavaScript, TypeScript
- **Frameworks:** React, FastAPI, Node.js
- **AI Models:** GPT-4o, Whisper, Stable Diffusion/Runway Gen-3, OpenAI TTS
- **Cloud Services:** AWS S3, GCP Video Intelligence API, Firebase
- **Databases:** MongoDB, PostgreSQL
- **Tools:** LangChain Agents, CrewAI, FFmpeg, Docker, Hugging Face

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- Docker (optional)
- FFmpeg

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/aniket-singh-01/saarthix.git
   cd saarthix
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   npm install
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the platform**
   ```bash
   python main.py
   ```

### Configuration

Update the configuration files in `environment/config/` to customize:
- Language preferences
- Content categories
- Output formats
- Regional settings

---

## 📊 Project Structure

```
saarthix/
├── environment/
│   ├── agents/          # Multi-agent system components
│   ├── config/          # Platform configuration files
│   ├── roles/           # Agent role definitions
│   └── communication/   # Inter-agent messaging
├── dataset/
│   ├── content/         # Educational content datasets
│   ├── voice_data/      # Voice synthesis training data
│   └── user_output/     # Generated video outputs
├── tools/
│   ├── CosyVoice/       # Voice synthesis engine
│   ├── fish-speech/     # Multilingual TTS
│   └── videorag/        # Video generation tools
└── assets/              # Platform assets and media
```

---

## 🌍 Supported Languages

### Primary Languages (20+ Official)
- Hindi, Bengali, Telugu, Marathi, Tamil
- Gujarati, Urdu, Kannada, Odia, Malayalam
- Punjabi, Assamese, Maithili, Santali, Kashmiri
- Nepali, Konkani, Sindhi, Dogri, Manipuri, Bodo

### Planned Additions
- 100+ regional dialects
- Sign language support
- Voice-to-video for low-literacy users

---

## 🎯 Use Cases & Examples

### Healthcare Awareness
```yaml
# Generate COVID-19 vaccination awareness video
content_type: healthcare
language: hindi
topic: "COVID-19 vaccination benefits"
target_audience: rural_communities
duration: 60_seconds
format: mobile_vertical
```

### Agricultural Advisory
```yaml
# Seasonal farming guidance
content_type: agriculture  
language: punjabi
topic: "Wheat sowing best practices"
target_audience: farmers
duration: 90_seconds
format: whatsapp_shareable
```

### Government Scheme Information
```yaml
# PM-KISAN scheme explanation
content_type: government
language: bengali
topic: "PM-KISAN application process"
target_audience: rural_farmers
duration: 120_seconds
format: social_media
```

---

## 🚧 Challenges We Overcame

- **Real-time translation** for Indian dialects and regional variations
- **GPU-heavy rendering** pipeline optimization for cost efficiency
- **Cultural sensitivity** in visual and audio content generation
- **Speed vs Quality** balance in production environments
- **Low-bandwidth optimization** for rural internet connectivity

---

## 🏆 Accomplishments

- ✅ Built a fully functional AI video pipeline in under 48 hours
- ✅ Delivered accurate multilingual scripts using open-source LLMs  
- ✅ Designed a no-code content upload system for NGOs and educators
- ✅ Created a scalable architecture for nationwide deployment
- ✅ Achieved 90%+ accuracy in regional dialect translation
- ✅ Optimized videos for 2G/3G networks (< 5MB for 60s video)

---

## 📚 What We Learned

- **Multi-agent AI systems** integration for creative workflows
- **Multilingual AI model** training and evaluation best practices  
- **Video generation optimization** for low-bandwidth environments
- **UX design importance** for diverse user bases (farmers to urban professionals)
- **Cultural sensitivity** in AI-generated content for Indian audiences
- **Scalable cloud architecture** for handling millions of users

---

## 🔮 What's Next for SaarthiX

### Short Term (3-6 months)
- 🎯 Expand to 50+ languages and dialects
- 📱 Build offline mode for rural areas
- 🤝 Partner with government agencies and NGOs
- 🎭 Add regional sign language support

### Medium Term (6-12 months)  
- ⚡ Real-time AI video generation for emergencies
- 🔗 Integration with WhatsApp Business API
- 📺 TV broadcast-ready content generation
- 🎓 Educational curriculum partnerships

### Long Term (1-2 years)
- 🌍 Expand to other developing nations
- 🤖 Advanced AI agents for specialized domains
- 📊 Analytics dashboard for content effectiveness
- 🏥 Integration with healthcare systems

---

## 🤝 Contributing

We welcome contributions from the community! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Code of conduct
- Development setup
- Submission process
- Language model training
- Testing procedures

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenAI** for GPT-4o and Whisper models
- **Hugging Face** for open-source model hosting
- **Indian Language Communities** for dialect training data
- **NGO Partners** for real-world testing and feedback
- **Government of India** Digital India initiative inspiration

---

## 📞 Contact & Support

- **Website:** [saarthix.ai](https://saarthix.ai)
- **Email:** support@saarthix.ai
- **Twitter:** [@SaarthiX](https://twitter.com/saarthix)
- **LinkedIn:** [SaarthiX](https://linkedin.com/company/saarthix)

### For Partnerships
- **NGOs & Educators:** partnerships@saarthix.ai
- **Government Agencies:** government@saarthix.ai  
- **Developers:** developers@saarthix.ai

---

**🚀 Built with ❤️ for Bharat's Digital Future**

*Making information accessible, one video at a time.*