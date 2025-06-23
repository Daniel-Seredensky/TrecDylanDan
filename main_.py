from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
import traceback

import os
import asyncio

from src.QA_Assistant.DocSelect import select_documents
from src.QA_Assistant.Assistant import get_or_create_assistant
from src.QA_Assistant.QuestionEval import assess_question
from pathlib import Path


async def _main():
    load_dotenv(dotenv_path=Path(".env"), override= True)
    api_key        = os.getenv("AZURE_OPENAI_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version    = os.getenv("AZURE_API_VERSION")
    print(api_key,azure_endpoint,api_version)
    try:
        doc = "'Kyle Rittenhouse (Kenosha Shooter) Bio, Age, Mother, Shooting, Arrest | Meforworld\nKyle Rittenhouse (Kenosha Shooter) Bio, Age, Mother, Shooting, Arrest\nKyle Rittenhouse is a 17 years old teen from Kenosha Wisconsin who was arrested and charged with killing two people during protests against the shooting of Jacob Blake. BY Admin\nKyle Rittenhouse Biography\nKyle Rittenhouse is a 17 years old teen from Kenosha Wisconsin who was arrested and charged with killing two people during protests against the shooting of Jacob Blake. Rittenhouse is said to have attended Lakes Community High School for a semester in the 2017-18 school year, according to Jim McKay, the superintendent of the school district. According to department letters, he also participated in cadet programs with both the Antioch Fire Department and the Grayslake Police Department. Kyle Rittenhouse Age\nRittenhouse is 17 years old as of 2020. Kyle Rittenhouse Mother\nAccording to the Washingtonpost, Rittenhouse is the son of Wendy Rittenhouse, a single mom and nurse’s assistant. He and his mom lived in an apartment complex beside a park in Antioch. Kyle Rittenhouse Photo\nKyle Rittenhouse Kenosha Shooting\nOn the night of August 25, 2020, Rittenhouse is alleged to have shot three protesters, two of whom later died. Police were alerted by onlookers that Rittenhouse, who is underage was walking around the street with a semiautomatic rifle slung around his neck. The two protesters who night that night were 26-year-old from Silver Lake, Wisconsin, and a 36-year-old from Kenosha. Governor’s Statement / Governor Tony Evers’ Statement\n“My heart breaks for the families and loved ones of the two individuals who lost their lives and the individual who was injured last night in Kenosha. We as a state are mourning this tragedy.” He went on to call the protests in Kenosha a reflection of the “pain, anguish and exhaustion of Black people in our state and country.” Kyle Rittenhouse Arrest\nRittenhouse was arrested on August 26, 2020, across the border in Antioch, Ill. As of the day he was arrested, Rittenhouse was being held without bond in Lake County. We will provide more details as the story develops. Post navigation\nPrevious: Previous post: Shemar Moore Bio, Age, Height, Parents, Siblings, Wife, Soul Train, Net Worth, Movies\nNext: Next post: Carolyn Boseman (Chadwick Boseman’s Mother) Bio, Age, Husband, Son’s Death, Net Worth\nCategories: Famous People in USA '"
        questions = "Is this article accurate?"
        client = AsyncAzureOpenAI(                       # or alias as shown above
            api_key        = os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version    = os.getenv("AZURE_API_VERSION"),
            timeout        = 30.0,
            max_retries    = 3,
        )
        assistant = await get_or_create_assistant(client)
        s = await assess_question(question= questions, document= doc, client= client, assistant_id= assistant)
        print(s)
    except Exception as e:
        print (traceback.print_exc())
    finally:
        await client.close()

def main():
    asyncio.run(_main())

if __name__ == "__main__":
    main()

        
