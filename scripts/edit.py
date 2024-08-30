#!/usr/bin/env python3
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

    ## list post folder
    post_candidates = os.listdir(lang_path / 'post')
    if len(post_candidates) == 0:
        print(f'No post found in {lang_path}')
        return
    for i, post in enumerate(post_candidates):
        print(f'{i+1}. {post}')
    post_idx = non_empty_input(f'Enter the post index', 1)
    post_idx = int(post_idx)
    
    ## edit the post
    post = post_candidates[post_idx-1]
    post_path = lang_path / 'post' / post / 'index.md'
    SHELL_POPEN(f'{DEFAULT_EDITOR} {post_path.as_posix()}')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
    finally:
        exit(0)

