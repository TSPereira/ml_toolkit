import os
from io import StringIO

from pdfminer.converter import TextConverter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.layout import LAParams


def convert_pdf_to_txt(path: str, codec: str = 'utf-8', max_pages: int = 0, password: str = '', caching: bool = True,
                       check_extractable: bool = True):
    """Usage of the pdfminer to convert a pdf file into text.

    :param string path: The path for the pdf file.
    :param string codec: default:'utf-8'. Codec to use on the device (TextConverter)
    :param int max_pages: default: 0. 0 doesn't limit the max number of pages
    :param string password: password to unlock pdf if needed
    :param bool caching: default: True. Check documentation of pdfminer.PDFPage.get_pages()
    :param bool check_extractable: default: True. Check documentation of pdfminer.PDFPage.get_pages()
    :return string:
    """

    rsrc_mgr = PDFResourceManager()
    ret_str = StringIO()
    la_params = LAParams()
    pagenos = set()

    device = TextConverter(rsrc_mgr, ret_str, codec=codec, laparams=la_params)
    interpreter = PDFPageInterpreter(rsrc_mgr, device)

    # Remove any extension the path might have and add the pdf extension
    path = os.path.splitext(path)[0] + '.pdf'

    with open(path, 'rb') as fp:
        for page in PDFPage.get_pages(fp, pagenos, maxpages=max_pages, password=password, caching=caching,
                                      check_extractable=check_extractable):
            interpreter.process_page(page)
        text = ret_str.getvalue()

    device.close()
    ret_str.close()
    return text
