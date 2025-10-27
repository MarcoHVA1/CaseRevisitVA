import requests

periods = [
    ("20210101", "20221231", "Maastricht_2021_2022.json"),
    ("20220101", "20231231", "Maastricht_2022_2023.json"),
    ("20230101", "20241231", "Maastricht_2023_2024.json"),
]

url = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens"
headers = {"Content-Type": "application/x-www-form-urlencoded"}

for start, end, filename in periods:
    data = f"start={start}&end={end}&stns=380&vars=ALL&fmt=json"
    r = requests.post(url, headers=headers, data=data)
    r.raise_for_status()
    with open(filename, "wb") as f:
        f.write(r.content)
    print(f"✅ Saved {filename}")
