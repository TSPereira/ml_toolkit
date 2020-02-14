from pdfminer.converter import TextConverter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from io import StringIO
from pdfminer.layout import LAParams


def convert_pdf_to_txt(path, codec='utf-8', max_pages=0, password='', caching=True, check_extractable=True):
    """Usage of the pdfminer to convert a pdf file into text.

    :param string path:
    :return string text:  The converted file in a single string.

    :param string path: The path for the pdf file.
    :param string codec: default:'utf-8'. Codec to use on the device (TextConverter)
    :param int max_pages: default: 0. 0 doesn't limit the max number of pages
    :param string password: password to unlock pdf if needed
    :param bool caching: default: True. Check documentation of pdfminer.PDFPage.get_pages()
    :param bool check_extractable: default: True. Check documentation of pdfminer.PDFPage.get_pages()
    :return string:
    """

    rsrc_mgr = PDFResourceManager()
    la_params = LAParams()
    pagenos = set()

    with StringIO() as ret_str:
        with TextConverter(rsrc_mgr, ret_str, codec=codec, laparams=la_params) as device:
            interpreter = PDFPageInterpreter(rsrc_mgr, device)

            path += '.pdf' if not path.endswith('.pdf') else ''
            with open(path, 'rb') as fp:
                pages = PDFPage.get_pages(fp, pagenos, maxpages=max_pages, password=password, caching=caching,
                                          check_extractable=check_extractable)
                for page in pages:
                    interpreter.process_page(page)

        text = ret_str.getvalue()
    return text
