import datetime
from typing import Any, List

from ..type import Argument, Plugin



class Holiday(Plugin):
    name = "Holiday"
    name_for_human = "HolidayQuery"
    description = "It can be used to query the public holiday status in various countries."
    arguments: List[Argument] = [
        Argument(name="countryCode", type="string", description="The ISO 3166-1 two-digit letter code of the country, for example, CN means China, GB means the United Kingdom", required=False),
        Argument(name="year", type="int", description="Which year's festival", required=False),
    ]

    def run(self, args: str) -> Any:
        params = super().run(args)
        countryCode = params["countryCode"] if "countryCode" in params else "CN"
        year = params["year"] if "year" in params else datetime.date.today().year

        url = f"https://date.nager.at/Api/v2/PublicHolidays/{year}/{countryCode}"
        import requests
        try:
            response = requests.get(url, timeout=5)
        except requests.exceptions.Timeout:
            return "Query failed"
        
        if response.status_code != 200:
            return "Query failed"
        return response.json()
