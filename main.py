from typing import List
import json

from src.pipeline import RAGPipeline, InfoExtractor
from src.test_prompt import imgs_to_text


RAG_PROMPT = """
Given the following context, format all the tables in it into json format with the instruction below:
1. first look for financial information in the text and then extracted them into json.
You will be referring this property to define the relationship between entities. 
NEVER create new financial infomration that aren't mentioned below. 
The input is OCR from a financial statement PDF.

The list after the key Table represents columns, and the keys in column lists are for rows representation
If there are texts between tables, represents them as a part of the json, with the key "text"

2. Try to find all tables in the text and format them into json
3. Do NOT create duplicate entities
4. Never input missing values
5. The values with brackets e.g "(123.123)" should be kept the same in the json, do not add brakets to values that does not have it
6. Property name should be enclosed in double quotes
7. If the property is the final property in a list, DO NOT add comma after it. It can causes errors
8. The output should be load by only call json.loads(output)
9. You can ignore the header and footer if there are    

====== EXAMPLE CONTEXT START ======
Table cookie sale price:
         Sale from 12/6 to 12/9    Sale from 12/6 to 12/9  
Cookie   10.000.000                2.000.000

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Praesent fringilla eros sed nisl ultrices blandit. 
Interdum et malesuada fames ac ante ipsum primis in faucibus. Sed ac mauris non odio aliquam aliquam nec non nisl. 
Mauris luctus lorem at euismod vehicula. Proin ligula justo, pellentesque eu orci in, laoreet dictum mauris

Table profit and sold product:
         2023               2024
Profit   2.2                2.7
Sold     2002               3421

====== EXAMPLE CONTEXT END ======

====== EXAMPLE OUTPUT START ======
{"info":{
    "Table 1": [
        {"Sale from 12/6 to 12/9": {"Cookie":"10.000.000"}}, 
        {"Sale from 12/6 to 12/9": {"Cookie":"2.000.000"}}
    ],
    "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Praesent fringilla eros sed nisl ultrices blandit. 
            Interdum et malesuada fames ac ante ipsum primis in faucibus. Sed ac mauris non odio aliquam aliquam nec non nisl. 
            Mauris luctus lorem at euismod vehicula. Proin ligula justo, pellentesque eu orci in, laoreet dictum mauris"
    "Table valuable year": [
        {"2023": {"Profit":"2.2", "Sold":"2002"}},
        {"2024": {"Profit":"2.7", "Sold":"3421"}}
    ]
    }
}
====== EXAMPLE OUTPUT END ======

Context: 

{{query}}
"""

INFO_PROMPT = """
You are a master in financial analysis and you are given a text to analyze.
This is an excerpt from a financial statement. You are required to extract the information about the company's
profit and revenue of the quarter of the financial statêmnt. The text is in Vietnamese. 

The output should be a json file with key is "profit" and "revenue". Inside the key should contains "value" and "source" that shows the source of information. 
If there is no information about the profit or revenue, the value should be null and the source should be null.

The Vietnamese clue words for profit and revenue can be "lợi nhuận" and "doanh thu" respectively.
You should include the unit of the value in the output, like " xxx.t triệu đồng" or " yy.zz tỷ đồng", based on the information given.
The example of output is:
{
    "profit": {
        "value": ...,
        "source": ...
    },
    "revenue": {
        "value": ...,
        "source": ...
    }
}

This is the data you need to analyze:
{{query}}

Output:

"""


class FinStateRead:
    def __init__(self, pdf_img_dir: str, rag_prompt: str, info_prompt: str):
        print("Converting images to text")
        self.documents = imgs_to_text(pdf_img_dir)
        with open('my_list.json', 'w') as f:
            json.dump(self.documents, f)

        # with open('my_list.json', 'r') as f:
        #     self.documents = json.load(f)

        print("Init pipelines")
        self.rag_pipeline = RAGPipeline(self.documents, rag_prompt)
        self.info_extractor = InfoExtractor(info_prompt)

        print("Done")

    def run(self, query: str) -> str:
        print("Running RAG pipeline")
        rag_output = self.rag_pipeline.run(query)
        import ipdb; ipdb.set_trace()

        print("Running InfoExtractor")
        info_output = self.info_extractor.run(rag_output)
        import ipdb; ipdb.set_trace()

        return info_output


if __name__ == "__main__":
    fin_state_read = FinStateRead(pdf_img_dir='/Users/bachle/Main/Code/Projects/fin_state_read/is/sample',
                                  rag_prompt=RAG_PROMPT,
                                  info_prompt=INFO_PROMPT)
    res = fin_state_read.run("What is the profit and revenue of the company in the last quarter?")
    import ipdb; ipdb.set_trace()