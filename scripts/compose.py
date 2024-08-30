#!/usr/bin/env python3
from datetime import datetime
import os
from pathlib import Path
import subprocess as sp

DEFAULT_LANG = 'zh-cn'
DEFAULT_EDITOR = 'code'

SHELL_POPEN = lambda x: sp.Popen(x, stdout=sp.PIPE, stderr=sp.PIPE, shell=True)

def non_empty_input(prompt, default=None):
    while True:
        if default:
            value = input(f'{prompt} (default: {default}): ')
        else:
            value = input(f'{prompt}: ')
        ##
        if value:
            return value
        elif default:
            return default

def main():
    ## list content folder for languages
    lang_candidates = os.listdir('content')
    lang = non_empty_input(f'Enter the language {lang_candidates}', DEFAULT_LANG)
    if lang not in lang_candidates:
        lang = 'en'
    lang_path = Path('content') / lang

    ## input basic info
    title = non_empty_input('Enter the title')
    description = non_empty_input('Enter the description', '')
    slug = non_empty_input('Enter the slug', title.lower().replace(" ", "-"))
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S+0800')

    ## list categories
    cat_candidates = os.listdir(lang_path / 'categories')
    categories = non_empty_input(f'Enter the categories {cat_candidates}', '')
    categories = map(lambda x: x.strip(), categories.split(','))
    categorie_text = ''.join([ f'  - {cat}\n' for cat in categories ])

    ## pinned post
    pinned = non_empty_input('Is this a pinned post? (y/N)', 'n')
    weight_text = 'weight: 0' if pinned.lower() == 'y' else ''

    front_matter = f'''---
title: {title}
description: {description}
slug: {slug}
date: {date}
image: 
categories:
{categorie_text}
tags: 
{weight_text}
---
'''

    index_file = lang_path / 'post' / slug / 'index.md'
    print(f'Preview: ({index_file.as_posix()})')
    print(front_matter)
    print()

    ## write to file
    confirm = non_empty_input('Confirm to write to file? (Y/n)', 'y')
    if confirm.lower() == 'y':
        Path(index_file.parent).mkdir(parents=True, exist_ok=True)
        with open(index_file, 'w') as f:
            f.write(front_matter)

    print(f'Opening the file in VSCode...')
    SHELL_POPEN(f'{DEFAULT_EDITOR} {index_file.as_posix()}')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
    finally:
        exit(0)
