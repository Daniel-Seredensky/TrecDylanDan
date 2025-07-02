from QuestionSetGenerator import run_question_set_pipeline
import json

if __name__ == "__main__":
    # Example doc_id from FixedSampleData.jsonl
    doc_id = "msmarco_v2.1_doc_17_795452723#14_866362073"
    # Output path in DerivedData/QuestionSets
    safe_doc_id = doc_id.replace('/', '_').replace('#', '_')
    output_path = f"DerivedData/QuestionSets/{safe_doc_id}.json"
    result = run_question_set_pipeline(doc_id, output_path=output_path)
    print("Result dictionary:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nWrote question set output to {output_path}") 