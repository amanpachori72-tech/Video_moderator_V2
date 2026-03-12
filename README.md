**🛡️ AI-Powered Video Content Moderator**


An end-to-end Machine Learning application designed to moderate video content by detecting Hate Speech and Offensive Language using Natural Language Processing (NLP).

**🚀 Overview**
Content moderation is a critical challenge for digital platforms. This project automates the process by extracting audio from video files, transcribing it into text, and using a trained ML classifier to categorize the sentiment. It features a unique dual-mode analysis (Automated & Manual) to ensure high reliability even in noisy environments.

📊 Dataset Summary: Hate Speech & Offensive Language
Source: Derived from the CrowdFlower / Davidson research dataset.

1. Data Composition
The dataset consists of approximately 24,000+ labeled rows collected from social media (Twitter). Each entry is categorized into three distinct classes:

Class 0 (Hate Speech): Language used to express hatred or encourage violence towards a group based on race, religion, sexual orientation, etc.

Class 1 (Offensive): Use of profanity, slurs, or derogatory language that may not necessarily constitute hate speech but is toxic for a general audience.

Class 2 (Neither/Safe): Clean, non-toxic conversational text.

2. Key Challenges Addressed
Class Imbalance: In the real world, "Safe" and "Offensive" comments far outnumber "Hate Speech." We addressed this by using Balanced Class Weights in the model to ensure the minority class (Hate Speech) was not ignored.

Contextual Ambiguity: Many words can be offensive in one context but not another. We used N-grams (1,2) during vectorization to capture two-word phrases, helping the model understand the difference between a single bad word and a descriptive sentence.

3. Pre-processing Pipeline
Before training, the raw text underwent:

Lowercasing: To ensure "Hate" and "hate" are treated equally.

Stop-word Removal: Removing common words (the, is, at) that don't add emotional value.

TF-IDF Vectorization: Converting text into numerical values based on the "importance" of a word across the entire dataset rather than just its frequency.

**🤖 Machine Learning Technical Summary**
1. The Model Architecture
Algorithm: Logistic Regression chosen for its high interpretability and efficiency in multi-class text classification.

Classification Strategy: One-vs-Rest (OvR), allowing the model to distinguish between three distinct boundaries: Hate Speech, Offensive, and Safe.

Hyperparameter Tuning: Applied class_weight='balanced' to penalize the misclassification of minority classes (Hate Speech) more heavily, correcting the inherent bias in social media datasets.

2. Feature Engineering (NLP Pipeline)
To convert raw text into a mathematical format the model can understand, we implemented a TF-IDF (Term Frequency-Inverse Document Frequency) pipeline:

Vectorization: We used a vocabulary of the top 5,000 features.

N-gram Analysis: Instead of just looking at single words (Unigrams), we used Bigrams (1,2). This allows the model to understand the difference between "bad" (offensive) and "not bad" (safe).

Preprocessing: Implemented automated lowercasing and stop-word removal to reduce noise and focus on "high-impact" tokens.

3. Audio-to-Text Integration
Signal Processing: Raw audio was normalized to a 16kHz sample rate and converted to Mono channel using Pydub to optimize it for the Speech-to-Text engine.

Transcription Engine: Utilized the Google Speech-to-Text API with language-specific tuning (en-IN) to handle Indian accents and localized slang effectively.

4. Decision Logic & Evaluation
Probability Estimation: The model doesn't just give a label; it calculates a Probability Distribution across all classes using the predict_proba function.

Confidence Scoring: The "Confidence Score" shown on the dashboard is the highest probability value from that distribution, providing transparency into how "sure" the model is about its verdict.

%% Technical Flowchart for AI Video Moderator
graph TD
    %% Define styles for clarity
    classDef process fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef data fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef storage fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef error fill:#ffcdd2,stroke:#c62828,stroke-width:2px;
    
**WorkFlow**

    %% 1. Input Stage
    Start(<strong>USER UPLOADS MP4 VIDEO</strong>) --> Upload[<center>Streamlit file_uploader</center>]
    Upload --> |Stores Temporary File| TempVid((<strong>temp_video.mp4</strong>))

    %% 2. Process - Audio Extraction & Conversion
    TempVid --> Extract{<strong>Extract Audio</strong><br/><em>MoviePy</em>}::process
    Extract --> |Raw WAV| RawAudio((<strong>raw_audio.wav</strong>))

    RawAudio --> Pydub{<strong>Audio Pre-processing</strong><br/><em>Pydub</em>}::process
    Pydub --> |1. Mono Conversion<br/>2. 16kHz Sampling| CleanAudio((<strong>clean_audio.wav</strong>))

    %% 3. Process - Speech-to-Text
    CleanAudio --> STT{<strong>Speech-to-Text</strong><br/><em>SpeechRecognition API</em>}::process
    STT --> |en-IN Language Setting| Transcript{<strong>Generate Transcript</strong>}::decision

    %% 4. Decision - Transcription Success?
    Transcript --> |Success| NLP{<strong>NLP Pipeline</strong><br/><em>TF-IDF Vectorizer</em>}::process
    Transcript --> |Failure / Silent / Noisy| Manual(<strong>MANUAL OVERRIDE TAB</strong>)::error
    Manual --> |User Inputs Text| NLP

    %% 5. Process - NLP & ML
    NLP --> |1. Lowercasing<br/>2. Stop-word Removal<br/>3. N-grams 1,2| Features((<strong>Feature Vector</strong>))

    Features --> LoadModel[<strong>LOAD TRAINED MODEL</strong><br/><em>moderator_model.pkl</em>]::storage
    LoadModel --> Predict{<strong>Predict Sentiment</strong><br/><em>Logistic Regression</em>}::process

    %% 6. Process - Probability & Decision
    Predict --> |calculate| Proba{<strong>predict_proba</strong>}::data
    Proba --> |Highest Value| ConfScore(<strong>Calculate Confidence %</strong>)::data

    %% 7. Output Stage
    ConfScore --> Result{<strong>DETERMINE VERDICT</strong>}::decision
    Result --> |Class 0| Hate[🔴 <strong>HATE SPEECH</strong>]
    Result --> |Class 1| Off[🟡 <strong>OFFENSIVE</strong>]
    Result --> |Class 2| Safe[🟢 <strong>SAFE</strong>]

    Hate --> Display(<strong>DISPLAY ON STREAMLIT DASHBOARD</strong>)::data
    Off --> Display
    Safe --> Display
    Display --> End(END)

    %% Add visual separators
    subgraph "FRONTEND"
        Upload
        Display
        Manual
    end
    
    subgraph "AUDIO & SPEECH-TO-TEXT"
        Extract
        Pydub
        STT
    end
    
    subgraph "MACHINE LEARNING"
        NLP
        LoadModel
        Predict
        Result
    end%%
  
  **Conclusion**

   The AI-Powered Video Content Moderator successfully demonstrates an end-to-end multimodal pipeline that bridges the gap between raw video data and automated sentiment analysis. By integrating MoviePy for audio extraction, Pydub for signal normalization, and a Logistic Regression model optimized with balanced class weights, the system effectively identifies toxic content while maintaining transparency through confidence scoring. The inclusion of a Manual Override feature further enhances the tool's real-world utility, ensuring reliability even in noisy environments where Speech-to-Text APIs might fail. This project serves as a robust prototype for automated digital safety, proving that a combination of machine intelligence and human-centric design can significantly streamline content moderation workflows.
   

    
