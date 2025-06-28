from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import asyncio
import json
import os
from pathlib import Path
from typing import List
import traceback

# ── third‑party ───────────────────────────────────────────────────────────────
import aiofiles                       # pip install aiofiles
from asyncinit import asyncinit       # pip install asyncinit
from dotenv import load_dotenv        # pip install python‑dotenv
from openai import AsyncAzureOpenAI   # pip install openai

# ── local project ────────────────────────────────────────────────────────────
from src.QA_Assistant.rate_limits import (
    openai_req_limiter,
    _global_tok_limiter,
    cohere_rerank_limiter
)
from src.QA_Assistant.bucket_monitor import BucketMonitor
from src.QA_Assistant.Assistant import get_or_create_assistant
from src.QA_Assistant.QuestionEval import assess_question

# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

# ── helpers ──────────────────────────────────────────────────────────────────
async def read_question_groups(json_path: Path) -> List[List[str]]:
    """Load *json_path* and return a list‑of‑lists of questions (one sub‑list
    per `"groups"` entry)."""
    async with aiofiles.open(json_path, "r", encoding="utf-8") as fp:
        data = json.loads(await fp.read())
    return [group.get("questions", []) for group in data.get("groups", [])]


# ── main orchestrator ────────────────────────────────────────────────────────
@asyncinit
class ContextProctor:
    """Runs `assess_question` in parallel batches and writes the combined
    context to ``CONTEXT_PATH`` from the environment."""

    _MAX_PARALLEL: int = 5

    async def __init__(
        self,
        *,
        template_path: Path,
        document: str,
        client: AsyncAzureOpenAI,
    ) -> None:
        self.document = document
        self.client = client
        self.context_path = Path(os.getenv("CONTEXT_PATH", "context.md")).expanduser()

        # Load template questions once
        self.question_groups: List[List[str]] = await read_question_groups(
            template_path
        )

        # Re‑use existing assistant or create a new one
        try:
            self.assistant_id = await get_or_create_assistant(self.client)
        except Exception:
            await self.client.aclose()
            raise 

    # ────────────────────────────────────────────────────────────────────────
    async def create_context(self) -> Path:
        """Call *assess_question* for every group (batched) and write results.

        Returns
        -------
        Path
            The file that was written.
        """
        tasks = [
            assess_question(
                question="\n".join(group),
                document=self.document,
                assistant_id=self.assistant_id,
                client=self.client,
            )
            for group in self.question_groups
        ]

        results = []
        try:
            # Throttle concurrency in batches of `_MAX_PARALLEL`
            for i in range(0, len(tasks), self._MAX_PARALLEL):
                batch = tasks[i : i + self._MAX_PARALLEL]
                results.extend(await asyncio.gather(*batch))
        finally:
            # Always close the client, even if gather() raised
            await self.client.close()

        # Persist the combined context
        async with aiofiles.open(self.context_path, "w", encoding="utf-8") as fp:
            for res in results:
                await fp.write(
                    f"Assistant status:\n{res['status']}\n\n"
                    f"Assistant answers:\n{res['content']}\n\n"
                )

        return self.context_path


async def _main():
    doc =  """
            "Are the Protest Movements Dead in Russia and Belarus?  - The Moscow Times\nNow is the time to support independent reporting from Russia!\nContribute Today\nopinion\nNord Stream 2 Is Far From a Done Deal\nThe U.S.-German agreement on the controversial gas pipeline is full of vague formulations and lacks concrete implementation mechanisms.\nBy Maria Shagina\nUpdated: July 27, 2021\nOn July 21, the U.S. and Germany signed an agreement to allow the completion of the controversial Nord Stream 2 gas pipeline.\nThe joint statement features a broad package of measures aimed at mitigating Ukraine’s security concerns and energy vulnerabilities while supporting the country on its way to decarbonization. The announced agreement is still far from a done deal. To become a grand bargain, it is still contingent on Ukraine’s consent. Despite all the assurances, commitments and support for Ukraine, Kiev is not buying them, as the agreement lacks binding security and economic obligations.\nThe U.S.-Germany agreement represents a shift from the purely economic narrative on Nord Stream 2 to a broader framework which marries security concerns, economic compensations and climate goals.\nHowever, the agreement did not manage to reconcile the divide between the U.S. and Germany’s security perceptions, while failing to engage with Ukraine properly. As a result, it is full of vague formulations and lacks concrete implementation mechanisms.\nnews\nPutin, Merkel 'Satisfied' With Near Completion of Nord Stream 2\nRead more\nTo mitigate Russia’s coercive use of energy and further aggressive acts against Ukraine, both the U.S. and Germany are ready to resort to new sanctions. Germany has pledged “to take action at the national level and press for effective measures at the European level, including sanctions, to limit Russian export capabilities to Europe in the energy sector.”\nThe willingness to impose new sanctions on Russia’s strategically important energy sector would definitely be a step up for the EU.\nSince 2014, the status quo has not changed and there has been little appetite for new restrictive measures.\nSanctions would also not be limited to the Russian gas sector and could extend to other economically relevant sectors.\nThis is an important addition as Russia’s use of geo-economic tools is often asymmetrical.\nThe U.S. could also impose new sanctions or rescind the already-in-place waivers on Nord Stream 2 AG and its CEO Matthias Warnig.\nThe text of the agreement leaves, however, plenty of questions unanswered: Did the U.S. and Germany manage to agree on what constitutes Russia’s use of energy as a weapon?\nThe disagreements on this subject can be traced to the Siberian pipeline crisis in the 1980s, when the U.S. and Germany failed to fundamentally agree on what constitutes energy dependency. Failing to draw lessons from the past would be a mistake. Leaving this question unanswered is consequential for the allies as to when to initiate the sanctions simultaneously.\nnews\nBiden: ‘Counter-Productive’ to Block Nord Stream 2\nRead more\nSecurity concerns\nIn the agreement, sanctions remain the main instrument to tackle Ukraine’s security concerns.\nKiev has been seeking stronger security guarantees to mitigate the threats posed by Nord Stream 2, but to no avail.\nLinking Nord Stream 2 to hard security issues such as the de-occupation of Ukrainian territories by Russia, discussing energy in the Normandy format or the delivery of weapons has been an unsuccessful negotiating tactic.\nBy raising the stakes high, Kiev has caught itself in the realpolitik scenario with little agency left to exercise.\nWhile sanctions can be an effective deterrent, their impact is limited if not properly embedded into a broader Russia strategy.\nBut there has been no change in Germany’s Russia policy, still deeply rooted in Ostpolitik. So it’s doubtful Russia will be deterred from violating the gas-transit contract by the threat of new sanctions alone.\nMoreover, with Merkel leaving her post in September, expanding economic sanctions will prove to be problematic. Will the new German chancellor be as effective and committed to forging sanctions coalitions on the EU level as Merkel has been since 2014?\nThe only additional security measure, on top of sanctions, is Germany’s commitment to increasing the capacity for reverse flows of gas to Ukraine, with the aim of shielding Ukraine completely from potential future attempts by Russia to cut gas supplies to the country. As it stands, the commitment is non-binding and depends on the energy market situation.\nOther security mechanisms, such as the so-called snap-back mechanism or the moratorium on the pipeline, are no longer discussed. In fact, German officials rejected a U.S. demand to include the snapback option, arguing that such state interference could be subject to a legal challenge.\nnews\nU.S. Will Not Sanction Main Nord Stream 2 Company – Reports\nRead more\nUkraine has been sceptical about the snapback option from the very beginning and does not believe that Berlin would be willing to cut off gas supplies via Nord Stream 2 if Russia violates the gas-transit agreement.\nGas-transit as a common denominator\nExtending the current gas-transit agreement beyond 2024 seems to be the common denominator Washington, Berlin and Kiev agree on.\nIn their joint statement, the U.S. and Germany emphasized the importance of Ukraine as a gas-transit country after 2024.\nThe main burden on delivering this measure will be on Germany. Berlin said it was committed “to utilize all available leverage to facilitate an extension of up to 10 years.”\nGermany will appoint a special envoy to support the negotiations which would begin no later than Sept. 1.\nIn fact, Merkel discussed the prolongation of the gas-transit contract with Putin on the same day the agreement was announced.\nHowever, the main questions concern the duration of the contract, the volumes and the tariffs. With Merkel leaving office in September, the window of opportunity for negotiating the extension is closing.\nDoes Merkel have enough time to convince Putin to prolong gas transit via Ukraine, when the primary goal of Nord Stream 2 is to diversify away from its neighbor?\nFor Ukraine, the extension of the contract is a necessary condition that addresses short-term vulnerabilities, but the lack of specificities triggers further insecurities and mistrust in Kiev.\nLong-term challenges of decarbonization\nFinally, to address Ukraine’s long-term challenges, the agreement aims to bolster Kiev’s decarbonization goals.\nThrough the Climate and Energy Partnership, the U.S. and Germany will support Ukraine’s energy transition, energy efficiency and energy security.\nWashington and Berlin pledged to establish a Green Fund, with expected $1 billion worth of investments. Germany will provide an initial contribution of at least $175 million to the fund, and appoint a special envoy with dedicated funding of $70 million. Germany pledged to support bilateral energy projects with Ukraine, especially in the field of renewables, hydrogen and the coal phase-out. The U.S. will provide assistance to Ukraine in market integration, regulatory reform, and renewables development. If Washington and Berlin properly deliver on their financial commitments, Ukraine could emerge much better prepared for energy transition than other emerging countries.\nIn the long term, pivoting away from its toxic energy dependency on Russia will help Ukraine to transform its energy system and strengthen its economy and national security. Kiev is, however, reluctant to accept the U.S. and Germany’s support in decarbonization plans as a mitigation against the Nord Stream 2 threats.\nUsing football terminology, Ukraine’s foreign minister said the game isn’t over yet, but has gone into overtime. The negotiations will continue and depend on whether the US and Germany can accommodate Ukraine’s perception of the pipeline as a security threat and what issues Ukraine will be willing to compromise on for such a highly sensitive issue.\nThe views expressed in opinion pieces do not necessarily reflect the position of The Moscow Times.\nMaria Shagina\nDr. Maria Shagina is a postdoctoral fellow at the Center for Eastern European Studies at the University of Zurich and a member of the Geneva International Sanctions Network @maria_shagina\nThe best of The Moscow Times, delivered to your inbox.\nRussia media is under attack.\nAt least 10 independent media outlets have been blocked or closed down over their coverage of the war in Ukraine.\nThe Moscow Times needs your help more than ever as we cover this devastating invasion and its sweeping impacts on Russian society.\nContribute today Maybe later\nopinion\nAre the Protest Movements Dead in Russia and Belarus?\nOne year on, repressive measures have squelched the protest movements that arose in Khabarovsk and Belarus, but that smoldering resentment has simply gone underground.\nBy Andrei Kolesnikov\nUpdated: July 27, 2021\nVyacheslav Prokofyev / TASS\nWhat do former Khabarovsk region Governor Sergei Furgal and Belarusian President Alexander Lukashenko have in common? To be sure, both have reputations as flawed individuals.\nThe difference, however, is that Furgal was voted into office by the people. Lukashenko was not. In both cases, the people — who now see themselves not as the vassals of unjust authority, but as citizens with rights — felt that their choice for officeholder had been stolen from them. That is why protesters in Khabarovsk and Belarus felt mutual solidarity. Their comradery was born of the fact that the civil society of both countries faced increasingly aggressive crackdowns by authoritarian regimes that had lost all pretense of civility.\nThe generality of Russians, who are usually wary of protests staged by the more advanced and educated residents of Moscow and St. Petersburg, felt a greater kinship with demonstrators in Khabarovsk.\nIn the eyes of average Russians — even those addicted to the mind-numbing drug of state-sponsored television propaganda — these were simply citizens like themselves whom the authorities had treated unfairly. And because the rallies took place far from Moscow, few believed that they were instigated by U.S. agents or supporters of opposition leader Alexei Navalny.\nIf anything, this gave the protests even greater appeal. As of Oct. 2020, according to the independent Levada Center, 47% of Russians expressed sympathy for the actions in Khabarovsk. By October, however, after the public’s attention had dissipated somewhat, that number had fallen to 43%.\nInterestingly, the Levada Center found that only 18% of Russians expressed the same support for protestors in “fraternal” Belarus, while 43% sided with Lukashenko.\nnews\nTens of Thousands Continue to Rally in Fresh Khabarovsk Protest\nRead more\nApparently, most Russians were still in thrall to the myth that order, cleanliness and justice reign in Belarus — thanks to Lukashenko. Russians who receive most of their information from state-controlled TV showed even less support for the Belarusian protesters, while those who primarily get their news on the Internet and social networks expressed greater support.\nIn short, relatively uninformed Russians fed by state propaganda saw a big difference between the protests in Khabarovsk and Belarus and failed to recognize that both groups were simply demanding that the authorities respect the will of the voters.\nThe Russian authorities monitored events in Khabarovsk closely. They understood intuitively that they could not crack down as severely on the ordinary citizens who took to the streets there as they did with the more “sophisticated hipsters” who fueled protests in Moscow. And even when the demonstrations quickly turned into strident anti-Putin rallies, the Kremlin waited.\nOnly in the fall of 2020 did the authorities begin to apply repressive measures directly, albeit without the same harshness they reserved for uppity college-types in the capital.\nThe Kremlin felt mystified by the Khabarovsk protests and treated them with great caution, just as it did the protracted protest against the government’s plans to build an enormous landfill in the remote town of Shies — an initiative that Moscow scrapped quietly and without publicity, in response to the civil society demand.\nThe Kremlin’s tactic of waiting out the Khabarovsk protesters has proven itself, at least for now.\nLeaders realize that no protest can last forever. Meanwhile, they have continued their case against Furgal, even as the number of protesters in the remote far eastern region continues to dwindle. What’s more, the Moscow opposition has switched its attention to something far more compelling: Navalny’s poisoning.\nnews\nBelarus Jails Students and Raids Media in Crackdown\nRead more\nBoth the Russian authorities and the civil society have followed the Belarusian protests with heightened interest, anticipating that the same thing could happen here during the presidential elections in 2024.\nut this theory got tested, at least indirectly, when Navalny — the main opponent of Putin and his regime — returned to Russia in January.\nThe Belarusian authorities took a tough and uncompromisingly repressive approach to the massive but peaceful protests in that country. The Russian authorities were just as recklessly harsh in suppressing protests in support of Navalny. Following acts of physical repression and arrests, they destroyed the opposition’s infrastructure and employed authoritarian legislation to liquidate numerous organizations of civil society and independent media outlets. They also placed the country’s electoral process under even tighter control than before.\nThe Russian authorities were not simply copying the behavior of their Belarusian counterparts: even without that inspiring example, greater repression was inevitable here because Putin’s regime has entered the stage of mature authoritarianism. Navalny’s poisoning only accelerated the transition to this more repressive phase.\nThe Belarusian example was important for another reason: it reinforced the Kremlin’s belief that it could only hold onto power through force. The time for subtle political games and negotiation has ended. Now starts an era in which the authorities commit fully to the use of harsh and uncompromising repression.\nPolitical scientist Tatiana Vorozheikina refers to the situation in Venezuela and Belarus as “the new resilience of authoritarian regimes.”\nAfter all, by all canons of historical and political science, these regimes should have fallen by now — and yet, they have survived. The same is true in Russia where, with white knuckles, the ruling elite cling desperately to power — meaning, to Putin — and will not give up either voluntarily. They are willing to use repression to achieve this, confident that the army, siloviki, intelligence agencies and the huge proportion of the population employed by the state or dependent on the state will remain loyal to their leaders.\nNevertheless, the protest movement will never disappear. It will go underground, like a subterranean fire. Nobody can predict where and when it will burst through to the surface, just as no one could have predicted the mass protests in Khabarovsk and Belarus.\nThe views expressed in opinion pieces do not necessarily reflect the position of The Moscow Times.\nAndrei Kolesnikov\nAndrei Kolesnikov is a senior associate and the chair of the Russian Domestic Politics and Political Institutions Program at the Carnegie Moscow Center. @AndrKolesnikov\nRead more about: Protest"
           """
    questions_template = Path("DebateAndReport/template_generator/output_1.json")
    client = AsyncAzureOpenAI(                       # or alias as shown above
            api_key        = os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version    = os.getenv("AZURE_API_VERSION"),
            timeout        = 30.0,
            max_retries    = 3,
        )
    try: 
        bucket_path = os.getenv("BUCKET_USAGE_PATH")
        bm = BucketMonitor(
            openai_req_bucket= openai_req_limiter,
            openai_tok_bucket= openai_tok_limiter,
            cohere_req_bucket= cohere_rerank_limiter,
            csv_path= f"{bucket_path}/bucket_usage.csv"
        )
        await bm.start()
        proctor = await ContextProctor (template_path= questions_template, client= client, document= doc)
        await proctor.create_context()
    except:
        traceback.print_exc()
    finally:
        await client.close()
        await bm.stop()

def main():
    asyncio.run(_main())

if __name__ == "__main__":
    main()
