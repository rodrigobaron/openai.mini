from typing import Any, List
from dotenv import load_dotenv
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper

from ..type import Argument, Plugin

load_dotenv()

wolfram = WolframAlphaAPIWrapper()


class Wolfram(Plugin):
    name = "wolfram"
    name_for_human = "Wolfram Alpha"
    description = "It can be used to calculate mathematical problems, solve equations and other mathematical operations, and can also be used to answer some time-sensitive questions, such as what is the current price of gold and where is the capital of the United States."
    arguments: List[Argument] = [
        Argument(name="query", type="string", description="The content to be calculated or solved, or the content to be queried", required=True)
    ]

    def run(self, args: str) -> Any:
        params = super().run(args)
        query = params["query"] if "query" in params else None
        if query is None:
            return None

        result = wolfram.run(query)
        parts = result.split("\nAnswer: ")
        return parts[-1].strip()
