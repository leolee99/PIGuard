import gdown

def pint_download(output_path='datasets/PINT.json'):
    file_id = '1OVkhLVrEXYkFGEGiwfk1LTUzDWMkQ1jz'

    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    gdown.download(url, output_path, quiet=False)
