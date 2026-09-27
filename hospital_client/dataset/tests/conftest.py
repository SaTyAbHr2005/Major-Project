
import pytest
import os
import shutil
import tempfile
import pandas as pd
from PIL import Image
import xlwt

@pytest.fixture(scope="function")
def comprehensive_image_dataset():
    temp_dir = tempfile.mkdtemp()
    
    formats_dir = os.path.join(temp_dir, "formats")
    os.makedirs(formats_dir)
    
    Image.new("RGB", (10, 10), color="red").save(os.path.join(formats_dir, "img1.jpg"))
    Image.new("RGB", (10, 10), color="blue").save(os.path.join(formats_dir, "img2.jpeg"))
    Image.new("L", (10, 10), color=128).save(os.path.join(formats_dir, "img3.png"))
    Image.new("RGB", (10, 10), color="green").save(os.path.join(formats_dir, "img4.bmp"))
    Image.new("RGB", (10, 10), color="yellow").save(os.path.join(formats_dir, "img5.tif"))
    Image.new("RGB", (10, 10), color="purple").save(os.path.join(formats_dir, "img6.tiff"))
    Image.new("RGBA", (10, 10), color=(255,0,0,128)).save(os.path.join(formats_dir, "img7.webp"))
    
    with open(os.path.join(formats_dir, "corrupt.jpg"), "wb") as f: f.write(b"")
    with open(os.path.join(formats_dir, "doc.pdf"), "w") as f: f.write("pdf")
    
    dir_a = os.path.join(temp_dir, "split_a") 
    os.makedirs(os.path.join(dir_a, "benign"))
    os.makedirs(os.path.join(dir_a, "malignant"))
    Image.new("RGB", (10,10), color="red").save(os.path.join(dir_a, "benign", "1.jpg"))
    Image.new("RGB", (10,10), color="blue").save(os.path.join(dir_a, "malignant", "2.jpg"))
    
    dir_b = os.path.join(temp_dir, "split_b") 
    for s in ["train", "test"]:
        for c in ["benign", "malignant"]:
            p = os.path.join(dir_b, s, c)
            os.makedirs(p)
            Image.new("RGB", (10,10)).save(os.path.join(p, "img.jpg"))
            
    dir_c = os.path.join(temp_dir, "split_c") 
    os.makedirs(os.path.join(dir_c, "minority"))
    os.makedirs(os.path.join(dir_c, "majority"))
    Image.new("RGB", (10,10)).save(os.path.join(dir_c, "minority", "1.jpg"))
    for i in range(10):
        Image.new("RGB", (10,10)).save(os.path.join(dir_c, "majority", f"{i}.jpg"))
        
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def comprehensive_tabular_dataset():
    temp_dir = tempfile.mkdtemp()
    
    csv_valid = os.path.join(temp_dir, "valid.csv")
    card_col = list(range(13))
    card_col[2] = card_col[1] # make row 2 and row 1 identical!
    
    df = pd.DataFrame({
        "num": [1, 2, 2, 2, None, 6, 7, 8, 9, 10, 11, 12, 13], 
        "cat": ["A", "B", "B", "A", "A", "C", "A", "B", "C", "A", "B", "C", "C"], 
        "text": ["hello world", "foo bar", "foo bar", "demo", "sample", "a", "b", "c", "d", "e", "f", "g", "h"],
        "constant_col": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "high_card_col": card_col
    })
    
    df.to_csv(csv_valid, index=False)
    with open(csv_valid, "r") as f:
        lines = f.readlines()
    lines[0] = lines[0].strip() + ",num\n"
    for i in range(1, len(lines)):
        lines[i] = lines[i].strip() + ",0\n"
    with open(csv_valid, "w") as f:
        f.writelines(lines)
    
    csv_empty = os.path.join(temp_dir, "empty.csv")
    with open(csv_empty, "w") as f: f.write("")
    
    csv_malformed = os.path.join(temp_dir, "malformed.csv")
    with open(csv_malformed, "w") as f: f.write("a,b,c\n1,2\n3,4,5,6")
    
    excel_file = os.path.join(temp_dir, "data.xlsx")
    with pd.ExcelWriter(excel_file) as w:
        pd.DataFrame({"x": [1,1]}).to_excel(w, sheet_name="S1", index=False)
        pd.DataFrame({"y": [2,2]}).to_excel(w, sheet_name="S2", index=False)
        
    xls_file = os.path.join(temp_dir, "data.xls")
    wb = xlwt.Workbook()
    ws = wb.add_sheet("Sheet1")
    ws.write(0, 0, "col1")
    ws.write(0, 1, "col2")
    ws.write(1, 0, 1)
    ws.write(1, 1, "A")
    wb.save(xls_file)
        
    yield temp_dir
    shutil.rmtree(temp_dir)

