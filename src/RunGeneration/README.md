# TREC-2025-DRAGUN Runs Generation

This package contains scripts and utilities for generating TREC-2025-DRAGUN runs from your existing pipeline in the required format.

## Files

- `generate_runs.py` - Main script to generate runs from topics
- `test_run_format.py` - Script to validate run format compliance
- `convert_to_run_format.py` - Convert existing pipeline output to run format
- `README.md` - This documentation file

## Prerequisites

1. **Environment Setup**: Make sure your `.env` file is configured with:
   ```bash
   OPENAI_API_KEY=your_openai_key
   AZURE_OPENAI_KEY=your_azure_key
   AZURE_OPENAI_ENDPOINT=your_azure_endpoint
   CONTEXT_PATH=path/to/context/file
   ```

2. **Dependencies**: All required packages should be installed in your virtual environment.

## Quick Start

### Option 1: Convert Existing Pipeline Output

If you already have pipeline output (like `japan_report.json`), convert it to TREC format:

```bash
# Convert your existing output
python src/RunGeneration/convert_to_run_format.py japan_report.json japan_run.json msmarco_v2.1_doc_japan_example

# Validate the output
python src/RunGeneration/test_run_format.py japan_run.json
```

### Option 2: Generate Runs from Topics File

To generate runs for all topics in the TREC topics file:

```bash
# Generate runs for all topics (this will take a while)
python src/RunGeneration/generate_runs.py trec-2025-dragun-topics.jsonl all_runs.jsonl

# Or test with just a few topics first
python src/RunGeneration/generate_runs.py trec-2025-dragun-topics.jsonl test_runs.jsonl --max-topics 3

# Validate the output
python src/RunGeneration/test_run_format.py test_runs.jsonl
```

## Output Format

The generated runs must follow this exact JSONL format:

```json
{
    "metadata": {
        "team_id": "SCIAI",
        "run_id": "SCIAI-run-example", 
        "topic_id": "msmarco_v2.1_doc_xx_xxxxx0",
        "type": "automatic",
        "use_starter_kit": 0
    },
    "responses": [
        {
            "text": "This is the first sentence.",
            "citations": [
                "msmarco_v2.1_doc_xx_xxxxxx1#x_xxxxxx3",
                "msmarco_v2.1_doc_xx_xxxxxx2#x_xxxxxx4"
            ]
        },
        {
            "text": "This is the second sentence.",
            "citations": []
        }
    ]
}
```

### Field Descriptions

#### Metadata Fields
- `team_id`: Unique identifier for the team (set to "SCIAI")
- `run_id`: Unique identifier for the run, specifying both team and method
- `topic_id`: The docid of the target news article (the topic)
- `type`: Indicates whether the run is "automatic" or "manual"
- `use_starter_kit`: Set to 1 if based on starter kit, or 0 if not

#### Response Fields
- `text`: One generated sentence in plaintext
- `citations`: List of up to 3 docids of segments that this sentence is based on

### Requirements
- All text fields in the responses list must together not exceed 250 words total
- Citations for a sentence may be empty
- Citation order does not matter
- Maximum 3 citations per response

## Step-by-Step Example

Here's a complete example workflow:

### 1. Test with a Small Sample

```bash
# Generate runs for just 2 topics to test
python src/RunGeneration/generate_runs.py trec-2025-dragun-topics.jsonl sample_runs.jsonl --max-topics 2

# Check the output
head -n 1 sample_runs.jsonl | python -m json.tool
```

### 2. Validate the Output

```bash
# Validate the generated runs
python src/RunGeneration/test_run_format.py sample_runs.jsonl
```

### 3. Generate Full Run

```bash
# Generate runs for all topics
python src/RunGeneration/generate_runs.py trec-2025-dragun-topics.jsonl final_runs.jsonl \
    --team-id SCIAI \
    --run-id SCIAI-final-run \
    --type automatic \
    --use-starter-kit 0
```

### 4. Final Validation

```bash
# Validate the final output
python src/RunGeneration/test_run_format.py final_runs.jsonl

# Check the format
head -n 1 final_runs.jsonl | python -m json.tool
```

## Usage

### Basic Usage

```bash
# Generate runs from the topics file
python src/RunGeneration/generate_runs.py trec-2025-dragun-topics.jsonl runs.jsonl

# Test with first 5 topics only
python src/RunGeneration/generate_runs.py trec-2025-dragun-topics.jsonl runs.jsonl --max-topics 5
```

### Advanced Usage

```bash
# Custom team and run IDs
python src/RunGeneration/generate_runs.py topics.jsonl output.jsonl \
    --team-id SCIAI \
    --run-id SCIAI-run-1

# Manual run type
python src/RunGeneration/generate_runs.py topics.jsonl output.jsonl \
    --type manual \
    --use-starter-kit 1

# Full example with all options
python src/RunGeneration/generate_runs.py trec-2025-dragun-topics.jsonl final_runs.jsonl \
    --team-id SCIAI \
    --run-id SCIAI-final-run \
    --type automatic \
    --use-starter-kit 0 \
    --max-topics 10
```

### Validation

```bash
# Test the format with a sample run
python src/RunGeneration/test_run_format.py sample

# Validate generated runs
python src/RunGeneration/test_run_format.py runs.jsonl
```

## Pipeline Integration

The `generate_runs.py` script integrates with your existing pipeline:

1. **Loads topics** from the JSONL file
2. **Generates reports** using your existing `ReportGenerator` and `ReportEvaluator`
3. **Splits reports** into individual sentences as responses
4. **Adds citations** (currently using placeholder logic - you should enhance this)
5. **Validates** the output format
6. **Writes** the results to a JSONL file

### Customization Points

You may want to customize these functions in `generate_runs.py`:

1. **`split_report_into_responses()`** - How to split the generated report into sentences
2. **Citation logic** - Currently uses placeholder citations, replace with actual citation generation
3. **Word counting** - Currently uses simple `split()` method, you might want more sophisticated counting

## Testing

### Test with Sample Data

```bash
# Generate a small test run
python src/RunGeneration/generate_runs.py trec-2025-dragun-topics.jsonl test_runs.jsonl --max-topics 2

# Validate the test output
python src/RunGeneration/test_run_format.py test_runs.jsonl
```

### Validate Format

The `test_run_format.py` script checks:

- ✅ Required metadata fields are present
- ✅ Field types are correct
- ✅ Response format is valid
- ✅ Word count is under 250 words
- ✅ Citations are properly formatted
- ✅ No more than 3 citations per response

## Important Notes

### Word Limit
- **Total word count must be ≤ 250 words** across all responses
- The scripts will warn you if this limit is exceeded
- You may need to adjust the sentence splitting logic if you hit this limit

### Citations
- **Maximum 3 citations per response**
- Citations can be empty arrays
- Current implementation uses placeholder citations - you should enhance this

### Team ID
- Set to **"SCIAI"** as specified
- Run ID should be unique and descriptive

## Example Output

Here's what a valid run entry looks like:

```json
{
    "metadata": {
        "team_id": "SCIAI",
        "run_id": "SCIAI-run-example",
        "topic_id": "msmarco_v2.1_doc_04_420132660",
        "type": "automatic",
        "use_starter_kit": 0
    },
    "responses": [
        {
            "text": "The Hawaiian pizza was invented by Sam Panopoulos in 1962 at his restaurant in Ontario, Canada.",
            "citations": ["msmarco_v2.1_doc_04_420132660#0_1234567890"]
        },
        {
            "text": "Panopoulos experimented with pineapple as a pizza topping, which was unusual at the time.",
            "citations": ["msmarco_v2.1_doc_04_420132660#1_2345678901"]
        },
        {
            "text": "The combination of ham and pineapple became popular and is now known worldwide as Hawaiian pizza.",
            "citations": []
        }
    ]
}
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all pipeline dependencies are installed
2. **Environment variables**: Check that your `.env` file is properly configured
3. **Memory issues**: Use `--max-topics` to limit processing for testing
4. **API rate limits**: The script includes retry logic, but you may need to adjust timeouts
5. **Word Count Exceeded**: Adjust the sentence splitting in `generate_runs.py`

### Error Messages

- `"Error: Topics file not found"` - Check the file path
- `"Error: Invalid responses"` - Check the response generation logic
- `"Error: Total word count exceeds 250 word limit"` - Adjust sentence splitting logic

## Files Created

After running the scripts, you'll have:

- `sample_runs.jsonl` - Test runs (JSONL format, one run per line)
- `final_runs.jsonl` - Final runs for submission
- `japan_run.json` - Example converted run (single JSON format)

## Submission

The `final_runs.jsonl` file is ready for submission to TREC-2025-DRAGUN. Each line contains one run entry in the required format.

## Next Steps

1. **Enhance citation generation** - Replace placeholder citations with actual segment-based citations
2. **Improve sentence splitting** - Use more sophisticated NLP techniques for better sentence boundaries
3. **Add quality checks** - Implement additional validation for response quality
4. **Optimize performance** - Consider parallel processing for large topic sets 