import json
from time import sleep
import os
from src.pipeline import RAGPipeline, LLMPipeline, DefineKey
from src.pdf_ocr import imgs_to_text, pdf_to_image
from tqdm import tqdm
import pickle

FORMAT_PROMPT = """
Given the following context, format all the tables in it into json format with the instruction below:
1. First look for financial information in the text and then extracted them into json.
You will be referring this property to define the relationship between entities. 
NEVER create new financial infomration that aren't mentioned below. 
The input is OCR from a financial statement PDF.

2. Try to find all tables in the text and format them into json
The list after the key Table represents columns, and the keys in column lists are for rows representation
If there are texts between tables, represents them as a part of the json, with the key "text"

- Do NOT create duplicate entities
- Never input missing values
- The values with brackets e.g "(123.123)" should be kept the same in the json, do not add brakets to values that 
does not have it
- Property name should be enclosed in double quotes
- If the property is the final property in a list, DO NOT add comma after it. It can causes errors.
- IF THE KEY IS NONE, USE "" INSTEAD OF REMOVING IT.
- If the text is like "549.730.301.393 1.305.277.451.910 1.442.461.944.861 (16.497.636.631) 13.052.962.004 (Many 
numbers in a row with no explanation), it seems like a table. You should format it as a table with the key "Table".

3. The output should be load by only call json.loads(output)
4. You can ignore the header and footer if there are  
5. If there is a mixture of text and table, format the json as the example below. The information in the table will not
be included in the text key, and the information in the text key will not be included in the table. All text should
be put after the table key. Please dont input the same information in both text and table.
6. You dont need to include ====== OUTPUT START ====== and ====== OUTPUT END ====== in the output
7. Try to find as many tables as possible. But dont forget to add the remaining text as a new key.
8. The input is the OCR version of a PDF, so if there is meaningless words, replace it with suitable words.
For example ChênhiệhChuyểnđối can be "Chênh lệch Chuyển đổi". 


====== EXAMPLE CONTEXT START ======
Table cookie sale price:
         Sale from 12/6 to 12/9    Sale from 12/9 to 12/12  
Cookie          10.000                      2.000
Snacks          5.000                       1.000

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Praesent fringilla eros sed nisl ultrices blandit. 
Interdum et malesuada fames ac ante ipsum primis in faucibus. Sed ac mauris non odio aliquam aliquam nec non nisl. 
Mauris luctus lorem at euismod vehicula. 

Table profit and sold product:
         2023               2024
Profit   2.2                2.7
Sold     2002               3421
          30                 72 
Proin ligula justo, pellentesque eu orci in, laoreet dictum mauris

====== EXAMPLE CONTEXT END ======

====== EXAMPLE OUTPUT START ======
{"info":{
    "Table cookie sale price": {
        "Sale from 12/6 to 12/9": {"Cookie":"10.000", "Snacks":"5.000"}, 
        "Sale from 12/9 to 12/12": {"Cookie":"2.000"}, "Snacks":"1.000"}
    },
    "Table profit and sold product": {
        "2023": {"Profit":"2.2", "Sold":"2002", "":"30"},
        "2024": {"Profit":"2.7", "Sold":"3421", "":"72"}
    },
    "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Praesent fringilla eros sed nisl ultrices blandit. 
            Interdum et malesuada fames ac ante ipsum primis in faucibus. Sed ac mauris non odio aliquam aliquam nec non nisl. 
            Mauris luctus lorem at euismod vehicula. Proin ligula justo, pellentesque eu orci in, laoreet dictum mauris"
}}
====== EXAMPLE OUTPUT END ======

Context: 

{{query}}
"""

SUMMARIZE_PROMPT = """You are a financial analyst and you are given a text to summarize.
This is an excerpt from a financial statement. You are required to summarize the information in it. The text is in 
Vietnamese. You only need to provide a short summary of the section, like what the text about or what information is 
included in the section. Dont add anything new. 
You should warp up as much information as possible, which separates pages to pages. Then the user can know what is 
the main content of different pages without reading all of them.

These summaries will be used in RAG system as a embeded document. The output should be a string start by "this 
section" or "this page" and then provide a short summary of the section.

Headers should be included. Rows and Columns of tabes must be mentioned.

====== EXAMPLE CONTEXT START ======
CÔNG TY CỔ PHẦN FPT
Số 10 phố Phạm Văn Bạch Báo cáo tài chính hợp nhất
Phường Dịch Vọng, Quận Cầu Giấy Cho kỳ hoạt động từ ngày 01 tháng 01 năm 2024
Thành phố Hà Nội, Việt Nam đến ngày 31 tháng 3 năm 2024

F _ `... #,
BẰNG CĂN ĐỐI KẾ TOÁN HỢP NHẬT
Tại ngày 31 tháng 3 năm 2024
MẪU SỐ B 01-DN/HN
Đơn vị: VND
¬- „ _„ Thuyết

TÀI SÂN Mã số minh 31/3/2024 31/12/2023
A. TÀI SẲN NGÃN HẠN 100 37.679.149.034.489 36.705.751.751.876
I.. Tiền và các khoản tương đương tiền 110 5 6.341.457.504.996 8.279.156.683.221
1. Tiền 111 5.411.920.365.956 5.975.127.685.903
2. Các khoản tương đương tiền 112 929.,537.139.040 2.304.028.997.318
II. Các khoản đầu tư tài chính ngắn hạn 120 6 18.230.159.723.426 16.104.205.358.010
1. Đầu tư nắm giữ đến ngày đáo hạn 123 18.230.159.723.426 16.104.205.358.010
II. Các khoản phải thu ngắn hạn 130 10.054.871.366.217 9.674.343.237.3544
1, Phải thu ngắn hạn của khách hàng 131 ví 9.393.764.058.957 9.057.647.206.985
2. Trả trước cho người bán ngắn hạn 132 628.462.570.207 482.074.732.731
S:PHAEIRALEHEG-IEHLBG.KGHOBL 134 190.429.912.043 176.770.894.412

hợp đồng xây dựng
4. Phải thu về cho vay ngắn hạn 135 657.520.000 515.430.000
5, Phải thu ngắn hạn khác 136 8 819.717.532.596 869.491.618.296
6. Dự phòng phải thu ngắn hạn khó đòi 137 9 (978.160.227.586) (912.156.645.080)
IV. Hàng tồn kho 140 10 1.856.403.641.469 1.593.411.075.233
1. Hàng tồn kho 141 1.996.597.949,120 1.724.956.924.671
2. Dự phòng giảm giá hàng tồn kho 149 (140.194.307.651) (131.545.849.438)
V. _ Tài sản ngắn hạn khác 150 1.196.256.798.381 1.054.635.398.068
1. Chi phí trả trước ngắn hạn 151 14 626.343.628.225 449.245.737.865
2. Thuế giá trị gia tăng được khấu trừ 152 80.443.466.850 528.984.574.991
3. Thuế và các khoản khác phải thu Nhà nước 153 18 89.469.703.306 76.405.085.212

2
ị

====== EXAMPLE CONTEXT END ======

====== EXAMPLE OUTPUT START ======
This page provides a detailed breakdown of short-term assets of FPT Corporation as of March 31, 2024, compared to 
December 31, 2023. It includes categories such as cash and cash equivalents, short-term financial investments, 
short-term receivables, inventory, and other short-term assets, with specific figures for each category and their 
subcategories.
====== EXAMPLE OUTPUT END ======

====== CONTEXT START ======
{{query}}
====== CONTEXT END ======

====== OUTPUT START ======

"""

RAG_PROMPT = """
You are a financial analyst and you are given a text as a context.
This is an excerpt from a financial statement. You are required to answer the question based on the context.
Remember to answer the question based on the context, do not add any new information.
The answer should be a part of the context, so the user can know where the answer is from.
The output should be in JSON format, with two keys "answer" and "source". The "answer" key contains the answer to the 
question, while the "source" key contains the part of the context that the answer is from.
The source is just a piece of information indicates the result, not the whole context.

The output should be load by only call json.loads(output) in Python.
* If the query ask about returning money, only return the number

====== EXAMPLE OUTPUT START ======
{
    "answer": "$1.2 million",
    "source": "FPT records $1.2 million in revenue for the last quarter."
}
====== EXAMPLE OUTPUT END ======

====== CONTEXT START ======
{{context}}
====== CONTEXT END ======

This is the question:
{{query}}
"""


class FinStateRead:
    def __init__(self,
                 pdf_img_dir: str,
                 format_prompt: str,
                 summary_prompt: str,
                 rag_prompt: str,
                 pre_ocred=None,
                 pre_ref_docs=None,
                 pre_docs=None,
                 save_ocred_path=None,
                 save_ref_docs_path=None,
                 save_docs_path=None,
                 azure_openai_key=None,
                 ):

        # Use pre_ref_docs and pre_docs if provided
        if pre_ref_docs is not None:
            self.ref_docs = pre_ref_docs
            self.documents = pre_docs

        else:
            # Cache the documents not to OCR next time
            print("Converting images to text")
            if pre_ocred:
                documents = pre_ocred

            else:
                if pdf_img_dir[-4:] == '.pdf':
                    pdf_to_image(pdf_img_dir, 'img_folder')
                    image_path = pdf_img_dir[:-4].split('/')[-1]
                    documents = imgs_to_text(os.path.join('img_folder', image_path))
                else:
                    documents = imgs_to_text(pdf_img_dir)

            # Save ocred documents
            if save_ocred_path:
                with open(save_ocred_path, 'wb') as f:
                    pickle.dump(documents, f)

            # Init extractor and summarizer
            print("Init pipelines...")
            self.def_key = DefineKey()
            self.info_extractor = LLMPipeline(format_prompt, azure_openai_key)
            self.summarizer = LLMPipeline(summary_prompt, azure_openai_key)

            # Reference documents
            print("Convert docs into json format...")
            if pre_ref_docs:
                self.ref_docs = pre_ref_docs
            else:
                self.ref_docs = []
                for doc in tqdm(documents):
                    self.def_key.change()
                    self.ref_docs.append(self.info_extractor.run(doc))
                    sleep(2)

            # Save ref document
            if save_ref_docs_path:
                with open(save_ref_docs_path, 'wb') as f:
                    pickle.dump(self.ref_docs, f)

            # Summarized Documents
            print("Summarize docs...")
            if pre_docs:
                self.documents = pre_docs
            else:
                self.documents = []
                for doc in tqdm(self.ref_docs):
                    self.def_key.change()
                    self.documents.append(self.summarizer.run(doc))
                    sleep(3)

            # Save docs
            if save_docs_path:
                with open(save_docs_path, 'wb') as f:
                    pickle.dump(self.documents, f)

        # Init rag pipeline
        print("Init RAG pipeline...")
        self.def_key = DefineKey()
        self.rag_pipeline = RAGPipeline(self.documents,
                                        self.ref_docs,
                                        rag_prompt)
        print("Done")

    def run(self, query: str) -> str:
        print("Running RAG pipeline...")
        rag_output = self.rag_pipeline.run(query)
        print(rag_output)

        return rag_output


if __name__ == "__main__":
    # Use ref_docs and docs from previous run
    with open('ref_docs_openai.pkl', 'rb') as file:
        ref_docs = pickle.load(file)

    with open('docs_openai.pkl', 'rb') as file:
        docs = pickle.load(file)

    fin_state_read = FinStateRead(pdf_img_dir='/Users/bachle/Main/Code/Projects/fin_state_read/img_folder/quy2_fpt_bctc',
                                  format_prompt=FORMAT_PROMPT,
                                  summary_prompt=SUMMARIZE_PROMPT,
                                  rag_prompt=RAG_PROMPT,
                                  pre_ref_docs=ref_docs,
                                  pre_docs=docs
                                  )

    with open('categories.txt', 'r') as f:
        cate_list = f.readlines()

    results = {}
    define_key = DefineKey()
    for cate in cate_list:
        # Change API key to avoid Gemini API limit
        define_key.change()
        response = fin_state_read.run(f"What is the {cate} of the company in the last quarter?")
        try:
            answer = json.loads(response)['answer']
        except:
            answer = response
        results[cate] = answer

        # Sleep to avoid API limit
        sleep(5)

    print(results)

    with open('result.json', 'w') as f:
        json.dump(results, f, indent=4)
