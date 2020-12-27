import requests
from bs4 import BeautifulSoup
from bs4.element import Tag


def get_page_html_BS(link):
    headers = {
        'User-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'snap Chromium/79.0.3945.79 Chrome/79.0.3945.79 Safari/537.36'}
    ret_html = requests.get(link, headers=headers).text
    return BeautifulSoup(ret_html, 'lxml')


def write_file(file_name, write_str):
    with open(file_name, 'a') as open_file:
        open_file.write(write_str)


def parce_source_1(save_file):
    main_link = 'http://pozdravok.ru'
    source = main_link + '/pozdravleniya/prazdniki/noviy-god/'
    response = get_page_html_BS(source)
    pages_link = []
    next_page_exists = True

    while next_page_exists:
        page_lins = response.find('div', {'class': 'pages'}).findAll('a')
        pages = [page['href'] for page in page_lins]

        for page in pages:
            pages_link.append(page)
        text_last_link = page_lins[-1].getText()
        if text_last_link != '...':
            next_page_exists = False
        else:
            response = get_page_html_BS(main_link + pages[-1])

    for page in pages_link:
        response = get_page_html_BS(main_link + page)

        pozdr = response.find('div',{'class':'content'}).findAll('p')
        for pozd in pozdr:
            text_pozd = ''
            for line_pozdr in pozd.contents:

                if type(line_pozdr) == Tag:
                    text_pozd = text_pozd + '\n'
                else:
                    text_pozd = text_pozd + line_pozdr
            text_pozd = text_pozd + '\n\n'
            write_file(save_file, text_pozd)
