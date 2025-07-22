import os
import openai
import json
from typing import List, Dict, Any, Optional

class ReportEvaluator:
    """
    Uses LLM evaluate a report, provide feedback, and generate IR questions.
    """
    def __init__(self, model: str = "gpt-4.1"):
        self.model = model
        self.client = openai.OpenAI()
    
    def evaluate(
        self,
        report: List[Dict[str, Any]],
        article_text: str,
        ir_context: Dict[str, Any],
        generator_comments: str = ""
    ) -> Dict[str, Any]:
        """
        Calls LLM to evaluate the report and return the evaluation contract.
        """
        prompt = self._build_prompt(report, article_text, ir_context, generator_comments)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a meticulous report evaluator for fact-checking and media literacy."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        # Try to extract JSON from the response
        result_text = response.choices[0].message.content if response.choices and response.choices[0].message and response.choices[0].message.content else ""
        try:
            # If the model returns markdown, strip it
            cleaned = result_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[len("```json"):].lstrip()
            if cleaned.startswith("```"):
                cleaned = cleaned[len("```"):].lstrip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].rstrip()
            result = json.loads(cleaned)
        except Exception as e:
            print("Failed to parse GPT-4.1 response as JSON:", e)
            print("Raw response:", result_text)
            result = {"error": "Failed to parse GPT-4.1 response as JSON.", "raw_response": result_text}
        return result

    def _build_prompt(self, report, article_text, ir_context, generator_comments):
        return f"""
You are an expert report evaluator for fact-checking and media literacy. Your job is to fill out the following evaluation contract for a fact-checking report.

**Report to evaluate (JSON array of sentences with citations):**
{json.dumps(report, indent=2)}

**Full article text:**
{article_text}

**IR Context (retrieved evidence, notes, or background):**
{json.dumps(ir_context, indent=2)}

**Report generator comments (about missing context, difficulties, etc):**
{generator_comments}

**Your tasks:**
1. **Grading:** Fill out a grading rubric as a JSON object with these keys (all required, scores 1-5, plus a detailed comment string):
   - coverage: How well does the report address the most important trustworthiness questions?
   - accuracy: Are all claims supported by the cited evidence? Any errors or unsupported statements?
   - citation_quality: Are citations relevant, sufficient, and diverse? Are all factual claims cited?
   - style: Is the report clear, concise, and objective?
   - prioritization: Does the report focus on the most important issues first?
   - completeness: Are there missing perspectives, unanswered questions, or gaps?
   - comments: A detailed, constructive summary of strengths and weaknesses (3-5 sentences).

2. **Message to Generator:** Write a short, actionable message (2-4 sentences) for the report generator. Focus on:
   - What to improve next (e.g., citation quality, missing perspectives, style).
   - How to address any gaps or weaknesses.
   - Avoid generic praise; be specific and constructive.

3. **IR Questions:** List up to 10 specific, non-redundant questions for further information retrieval. These should:
   - Address gaps, missing evidence, or unanswered high-priority questions.
   - Be clear, focused, and actionable.
   - Avoid repeating questions the generator already flagged as unanswerable.

**Output contract:**
Return a single valid JSON object with these fields:
- grading: (object, as above)
- message_to_generator: (string)
- ir_questions: (array of up to 10 strings)

**STRICT REQUIREMENTS:**
- Output only valid JSON. Do NOT include markdown, explanations, or extra text.
- Fill every required field, even if you must explain why a score is low or a field is empty.
- Do not hallucinate information. Only use the provided report, article, and IR context.
- If you cannot fill a field, explain why in the comments/message.

**Final checklist before output:**
- [ ] Output is a single valid JSON object, no markdown or extra text.
- [ ] All rubric fields are present and scored 1-5.
- [ ] Comments and message are specific, actionable, and grounded in the input.
- [ ] IR questions are clear, non-redundant, and address real gaps.

Begin your evaluation now.
"""

# Example usage
if __name__ == "__main__":
    example_report = [
        {
            "text": "Japan plans to discharge treated radioactive water from the Fukushima nuclear plant into the Pacific Ocean within two years.",
            "citations": ["msmarco_v2.1_doc_52_43359733#3_56562483"]
        },
        {
            "text": "The water will be filtered and diluted to safe levels before release, according to Japanese officials.",
            "citations": ["msmarco_v2.1_doc_52_43359733#3_56562483"]
        },
        {
            "text": "The Advanced Liquid Processing System (ALPS) is used to remove 62 types of radionuclides, except tritium, from the water.",
            "citations": ["msmarco_v2.1_doc_52_43359733#3_56562483"]
        },
        {
            "text": "Tritium is diluted to one-fortieth of Japan's drinking water standards before being released.",
            "citations": ["msmarco_v2.1_doc_52_1220313999#8_2434872501"]
        },
        {
            "text": "The International Atomic Energy Agency supports the plan as scientifically reasonable.",
            "citations": ["msmarco_v2.1_doc_52_43359733#3_56562483"]
        },
        {
            "text": "Critics, including neighboring countries and local fishermen, oppose the plan due to environmental concerns.",
            "citations": ["msmarco_v2.1_doc_52_43359733#3_56562483"]
        },
        {
            "text": "China and South Korea have expressed strong opposition, calling the decision irresponsible.",
            "citations": ["msmarco_v2.1_doc_52_43359733#3_56562483"]
        },
        {
            "text": "The release of treated water is consistent with international nuclear industry practices.",
            "citations": ["msmarco_v2.1_doc_52_43359733#3_56562483"]
        },
        {
            "text": "There are calls for independent monitoring to verify the safety of the discharged water.",
            "citations": ["msmarco_v2.1_doc_18_1439899514#2_1621043085"]
        }
    ]
    article_text = """ Japan says it will dump radioactive water from crippled Fukushima nuclear plant into the Pacific, sparking protests - CBS News
World
Protests as Japan says it will dump radioactive water from crippled Fukushima nuclear plant into the Pacific
By Lucy Craft
April 13, 2021 / 7:07 AM / CBS News
Tokyo — Japan said Tuesday that it would start discharging treated radioactive water from the crippled Fukushima nuclear power plant into the Pacific Ocean within two years. Officials in Tokyo said the water would be filtered and diluted to safe levels first, but many residents remain firmly opposed to the plan.
Protesters gathered outside Prime Minister Yoshihide Suga's residence in downtown Tokyo to denounce the government's decision.
More than a million tons of contaminated water is currently being stored at the Fukushima power plant in a massive tank farm big enough to fill 500 Olympic-sized swimming pools. The wastewater comes from water pumped in to cool the plant's damaged reactors and also rain and groundwater that seeps into the facility, which was seriously damaged by the 2011 earthquake and subsequent tsunami that ravaged Japan's northeast coast.
The unit three reactor building and storage tanks for contaminated water at the Tokyo Electric Power Company's (TEPCO) Fukushima Daiichi nuclear power plant in Okuma, Fukushima prefecture, Japan,   February 3, 2020.KAZUHIRO NOGI/AFP/Getty
The government says it has simply run out of room to store all the water. The plan to dump the water into the ocean first came to light in the autumn of last year, when Japanese news outlets cited anonymous officials as saying the decision had been taken.
"We can't postpone a decision on the plan to deal with the... processed water, to prevent delays in the decommission work of the Fukushima Daiichi nuclear power plant," Chief Cabinet Secretary Katsunobu Kato said in October 2020, without commenting directly on the plan or its timing.
On Tuesday, Suga said that after years of study, his scientific advisors had concluded that ocean discharge was the most feasible way to cope with the surplus of contaminated water.
"The International Atomic Energy Agency also supports this plan as scientifically reasonable," he said.
But the decision to dump Fukushima wastewater into the ocean has drawn fire from neighboring Asian countries and local fishermen along Japan's coast.
China called the decision "extremely irresponsible," and South Korea summoned the Japanese ambassador in Seoul over the matter.
Japan plans to release wastewater into ocean
01:59
"They told us that they wouldn't release the water into the sea without the support of fishermen," Kanji Tachiya, who leads a local cooperative of fisheries in Fukushima, told national broadcaster NHK ahead of the announcement on Tuesday. "We can't back this move to break that promise and release the water into the sea unilaterally."
Critics, including Greenpeace nuclear specialist Shaun Burnie, argue that Japan should continue storing the wastewater near the stricken Fukushima plant.
"Deliberately discharging and contaminating the Pacific Ocean after decades of contamination already from the nuclear industry and nuclear weapons testing is just not acceptable," he said.
The actual release of water from the Fukushima plant will take decades to complete. Critics have called on Japan's government to at least ensure that independent monitoring is in place to verify the level of radiation in the discharged water is safe for the environment.
In:
fukushima daiichi nuclear disaster
First published on April 13, 2021 / 4:53 AM
© 2021 CBS Interactive Inc. All Rights Reserved."""
    ir_context = {}
    generator_comments = "Could not find context for alternative viewpoints."
    evaluator = ReportEvaluator()
    result = evaluator.evaluate(example_report, article_text, ir_context, generator_comments)
    print(json.dumps(result, indent=2))
