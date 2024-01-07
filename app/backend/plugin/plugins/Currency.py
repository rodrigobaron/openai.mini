from typing import Any, List

from ..type import Argument, Plugin


apiVersion = "1"

class Currency(Plugin):
    name = "Currency"
    name_for_human = "Currency"
    description = "It can be used to perform exchange rate conversion, such as what is the exchange rate of RMB to US dollar."
    arguments: List[Argument] = [
        Argument(name="from_code", type="string", description="ISO 4217 three-letter code of exchange rate, for example, CNY means Renminbi, USD means United States dollar", required=True),
        Argument(name="to_code", type="string", description="ISO 4217 three-letter code of exchange rate, for example, CNY means Renminbi, USD means United States dollar", required=True),
        Argument(name="date", type="string", description="The exchange rate on that day must be in the date format of `YYYY-MM-DD`. If it is empty, it means querying the latest exchange rate.", required=False),
    ]

    def run(self, args: str) -> Any:
        params = super().run(args)
        from_code = params["from_code"] if "from_code" in params else "CNY"
        to_code = params["to_code"] if "to_code" in params else "USD"
        date = params["date"] if "date" in params else "latest"
        pattern_str = r'^\d{4}-\d{2}-\d{2}$'
        import re
        date = date if re.match(pattern_str, date) is not None else "latest"
        url = f"https://cdn.jsdelivr.net/gh/fawazahmed0/currency-api@{apiVersion}/{date}/currencies/{from_code.lower()}/{to_code.lower()}.json"
        import requests
        try:
            response = requests.get(url, timeout=5)
        except requests.exceptions.Timeout:
            return "Query failed"
        
        if response.status_code != 200:
            return "Query failed"
        return response.json()
