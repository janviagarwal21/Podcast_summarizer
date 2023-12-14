INTRODUCTION 
Our project, titled "Podcast Summarization Model," was a collaborative effort undertaken as part of our AI 3106: Foundations of NLP course at Ecole Centrale School of Engineering, Hyderabad. Our team, comprised of Harshveer Singh Thind, Srija Reddy K, A. Siddharth Reddy, TanmayiSri V, and Janvi Agarwal, set out to design an innovative solution for summarizing podcasts,which we approached by Speech to text analysis followed by text summarization. 

We chose the podcast summarization problem due to its relevance in light of the increasing popularity of podcasts and the apparent gaps in existing openly available summarization tools. We wanted to explore new avenues and contribute to the development of a more refined solution. 

APPROACH 
The approach here leverages Google's Speech Recognition models and BART (Bidirectional and Auto-Regressive Transformers) model from Hugging Face. The process included podcast audio segmentation using Pydub, speech-to-text conversion through Google Web Speech API, and summarization with the BART model. 

In this case, we have used a distilled BART model with a CNN decoder, with "sshleifer/distilbart-cnn-12-6" model architecture and configuration. 

TESTING ENVIRONMENT 
We utilized Google Colab, Notepad++, and Jupyter Notebook as our primary development environments. The report details the breakdown of our code for audio-to-text conversion, podcast segmentation, and the summarization model. 

EVALUATION 
For evaluation, we employed the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) package to collectively assess the quality of our summarization model. The discussion of ROUGE scores, including precision, recall, and F1 scores, provided valuable insights into how well our system-generated summary aligned with the reference summary. 

SETUP 
Google Colab Notebook comes with all the necessary packages, except the following that need to be installed with a ! pip install statement 
-SpeechRecognition 
-pydub 
-rouge 

RUN- 
A-Audio to text 
1-Segment the large audio file  
large_audio_file_path = "/content/scotus-cotus.wav" 
output_folder = "/content/audio_segmentsss" 
# Split the audio file into segments 
split_audio(large_audio_file_path, output_folder) 

2- Sort the segment files (the audio files in the output folder are unordered) 
sorted_audio_files = sorted(audio_files, key=lambda x: int(x.split('_')[1].split('-')[0])) 
# Iterate through the sorted audio files 
for audio_file in sorted_audio_files: 
    # Process the sorted audio files as needed 
    print(audio_file)  # Replace this with your code to use the sorted files 

3- Use the audio_to_text function to convert the audio to text 

B-Summarization 

1-Create a summarization pipeline 
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6") 

2-Function to generate summary for each article 
def generate_summary(article): 
    # Split the article into chunks of 1000 tokens 
    max_chunk_length = 100 
    chunks = [article[i:i + max_chunk_length] for i in range(0, len(article), max_chunk_length)] 
    # Generate summaries for each chunk 
    summaries = [summarizer(chunk, max_length=100, min_length=10, do_sample=False)[0]['summary_text'] for chunk in chunks] 

3-Combine summaries of each chunk 
    return " ".join(summaries) 

 

 
