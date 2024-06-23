import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
import random
import string
from tqdm.asyncio import tqdm
import time

def generate_random_name(length=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))

async def get_love_percentage(session, name1, name2, max_retries=7):
    url = f"https://www.calculator.net/love-calculator.html?cnameone={name1}&x=Calculate&cnametwo={name2}"
    retries = 0
    while retries < max_retries:
        try:
            async with session.get(url) as response:
                text = await response.text()
                soup = BeautifulSoup(text, 'html.parser')
                try:
                    percentage = soup.find_all('h1')[1].text.split("%")[0].strip()
                    return {'name1': name1, 'name2': name2, 'percentage': int(percentage)}
                except (IndexError, ValueError):
                    return None
        except (aiohttp.ClientError, aiohttp.http_exceptions.HttpProcessingError) as e:
            print(f"Request failed for {name1} and {name2}, retrying... ({retries+1}/{max_retries})")
            retries += 1
            await asyncio.sleep(2 ** retries)  # Exponential backoff
    print(f"Failed to fetch data for {name1} and {name2} after {max_retries} retries.")
    return None

async def main():
    name_pairs = [(generate_random_name(), generate_random_name()) for _ in range(10000)]  # Change 10000 to the desired number of pairs
    data = []

    async with aiohttp.ClientSession() as session:
        tasks = [get_love_percentage(session, name1, name2) for name1, name2 in name_pairs]
        
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Collecting data"):
            result = await task
            if result is not None:
                data.append(result)

    df = pd.DataFrame(data)
    df.to_csv('love_calculator_data.csv', index=False)
    print("Data collection complete and saved to love_calculator_data.csv")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
