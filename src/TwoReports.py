#!/usr/bin/env python3
"""
TwoReports.py - Generate two question sets from raw text using QuestionSetGenerator
"""

from QuestionGeneration.QuestionSetGenerator import run_question_set_pipeline

def generate_two_reports():
    # First document
    document_1 = """
    Denmark launches children's TV show about man with giant penis | Denmark | The Guardian
Denmark
This article is more than 1 year old
Denmark launches children's TV show about man with giant penis
Critics condemn idea of animated series about a man who cannot control his penis, but others have backed it
Still from the first episode of John Dillermand. Photograph: DRTV
Helen Russell in Jutland
Wed 6 Jan 2021 06.43 EST
Last modified on Tue 21 Dec 2021 10.24 EST
John Dillermand has an extraordinary penis. So extraordinary, in fact, that it can perform rescue operations, etch murals, hoist a flag and even steal ice-cream from children.
The Danish equivalent of the BBC, DR, has a new animated series aimed at four- to eight-year-olds about John Dillermand, the man with the world’s longest penis who overcomes hardships and challenges with his record-breaking genitals.
Unsurprisingly, the series has provoked debate about what good children’s television should – and should not – contain.
Sesame Street creates Rohingya Muppets to help refugee children
Read more
Since premiering on Saturday, opponents have condemned the idea of a man who cannot control his penis. “Is this really the message we want to send to children while we are in the middle of a huge #MeToo wave?” wrote the Danish author Anne Lise Marstrand-Jørgensen.
The show comes just months after the TV presenter Sofie Linde kickstarted Denmark’s #MeToo movement.
Christian Groes, an associate professor and gender researcher at Roskilde University, said he believed the programme’s celebration of the power of male genitalia could only set equality back. “It’s perpetuating the standard idea of a patriarchal society and normalising ‘locker room culture’ … that’s been used to excuse a lot of bad behaviour from men. It’s meant to be funny – so it’s seen as harmless. But it’s not. And we’re teaching this to our kids.”
Erla Heinesen Højsted, a clinical psychologist who works with families and children, said she believed the show’s opponents may be overthinking things. “John Dillermand talks to children and shares their way of thinking – and kids do find genitals funny,” she said.
“The show depicts a man who is impulsive and not always in control, who makes mistakes – like kids do, but crucially, Dillermand always makes it right. He takes responsibility for his actions. When a woman in the show tells him that he should keep his penis in his pants, for instance, he listens. Which is nice. He is accountable.”
Højsted conceded the timing was poor and that a show about bodies might have considered depicting “difference and diversity” beyond an oversized diller (Danish slang for penis; dillermand literally means “penis-man”). “But this is categorically not a show about sex,” she said. “To pretend it is projects adult ideas on it.”
DR, the Danish public service broadcaster, has a reputation for pushing boundaries – especially for children. Another stalwart of children’s scheduling is Onkel Reje, a popular figure who curses, smokes a pipe and eschews baths – think Mr Tumble meets Father Jack. A character in Gepetto News made conservatives bristle in 2012 when he revealed a love of cross-dressing. And Ultra Smider Tøjet (Ultra Strips Down) caused outrage in 2020 for presenting children aged 11-13 with a panel of nude adults, but, argues Højsted, such criticism was unjustified.
“What kind of culture are we creating for our children if it’s OK for them to see ‘perfect’ bodies on Instagram – enhanced, digitally or cosmetically – but not ‘real bodies’?” she said.
DR responded to the latest criticism by saying it could just as easily have made a programme “about a woman with no control over her vagina” and that the most important thing was that children enjoyed John Dillermand.
Topics
Denmark
Children's TV
Children
#MeToo movement
Europe
Gender
news
Reuse this content
    """
    
    # Second document
    document_2 = """
    Japan says it will dump radioactive water from crippled Fukushima nuclear plant into the Pacific, sparking protests - CBS News
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
© 2021 CBS Interactive Inc. All Rights Reserved.
    """
    
    # Generate the first question set
    run_question_set_pipeline(
        document_text=document_1,
        output_path="TwoReports/1001.json"
    )
    
    # Generate the second question set
    run_question_set_pipeline(
        document_text=document_2,
        output_path="TwoReports/1002.json"
    )

if __name__ == "__main__":
    generate_two_reports() 