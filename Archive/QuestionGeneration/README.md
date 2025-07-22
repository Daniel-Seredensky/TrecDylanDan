# QuestionGeneration

This folder contains tools and scripts for generating and extracting question sets from documents.

## Contents

- **QuestionSetGenerator.py**  
  Main module for generating question sets from documents using language models. Supports batching and async processing.

- **SummaryExtractor.py**  
  Extracts relevant document context and credibility assessments for each question.

- **example_run_question_set.py**  
  Example script demonstrating how to run the question set generation pipeline.

## Usage

1. Use `QuestionSetGenerator.py` to generate question sets from your documents.
2. Use `SummaryExtractor.py` to extract context and credibility for specific questions.
3. See `example_run_question_set.py` for a sample workflow.

## Notes

- Make sure required dependencies are installed (see project root for requirements).
- Designed for integration with larger document analysis pipelines. 