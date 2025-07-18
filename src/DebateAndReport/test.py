import requests
import os

# API Configuration
try:
    api_key = os.environ["LANGFLOW_API_KEY"]
except KeyError:
    raise ValueError("LANGFLOW_API_KEY environment variable not found. Please set your API key in the environment variables.")

url = "http://127.0.0.1:7860/api/v1/run/9074c506-c21b-43ae-9c2b-5339cadd4164"  # The complete API endpoint URL for this flow

# Request payload configuration
payload = {
    "output_type": "text",
    "input_type": "text",
    "input_value": """Denmark launches children's TV show about man with giant penis | Denmark | The Guardian
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
Reuse this content"""
}

# Request headers
headers = {
    "Content-Type": "application/json",
    "x-api-key": api_key  # Authentication key from environment variable
}

try:
    # Send API request
    response = requests.request("POST", url, json=payload, headers=headers)
    response.raise_for_status()  # Raise exception for bad status codes

    # Print response
    print(response.text)

except requests.exceptions.RequestException as e:
    print(f"Error making API request: {e}")
except ValueError as e:
    print(f"Error parsing response: {e}")