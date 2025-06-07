import re

austen_contents = [
        'PERSUASION',
        'NORTHANGER ABBEY',
        'MANSFIELD PARK',
        'EMMA',
        'LADY SUSAN',
        'LOVE AND FRIENDSHIP',
        'PRIDE AND PREJUDICE AND OTHER EARLY WORKS',
        'SENSE AND SENSIBILITY',
    ]

def find_start(work_name: str, text: str) -> int:
    """
    Find the start of a work in the text. Note that the first occurence is in the contents.
    """
    title = re.findall(f'{work_name.upper()}', text)
    if title:
        start_index = text.index(title[0])
        # Now search again for the second occurrence of the work name
        second_idx = text.index(title[0], start_index + len(title[0]))
        return second_idx
    else:
        raise ValueError(f"Work '{work_name}' not found in the text.")
    
def find_end(work_name: str, contents: list, text: str) -> int:
    """
    Find the end of a work in the text by looking for the next work in the contents.
    """
    next_work = contents[contents.index(work_name) + 1] if contents.index(work_name) + 1 < len(contents) else None
    if next_work:
        end_index = find_start(next_work, text)
        return end_index
    else:
        return len(text)  # If it's the last work, return the end of the text

def extract_work(work_name: str, contents: list, text: str) -> str:
    """
    Extract the text of a specific work by Jane Austen from the full text.
    """
    start_index = find_start(work_name, text)
    end_index = find_end(work_name, contents, text)
    return text[start_index:end_index].strip()

def extract_chapter(work: str, chapter_number: int = 1) -> str:
    """
    Extract a specific chapter from a work by Jane Austen.
    """
    chapters = re.split(r'\n\s*Chapter \d+\s*\n', work)
    if chapter_number <= len(chapters):
        # Chapter 0 is intro/preface which we skip
        return chapters[chapter_number].strip()
    else:
        raise ValueError(f"Chapter {chapter_number} does not exist in the work.")


def read_austen_chapter(work_name: str, chapter_number: int = 1, contents: list = austen_contents) -> str:
    """
    Read a specific work by Jane Austen and return the specified chapter.
    """
    with open('data/jane_austen_complete.txt', 'r', encoding='utf-8') as file:
        full_text = file.read()
    work_text = extract_work(work_name, contents, full_text)
    del full_text  # Free memory
    return extract_chapter(work_text, chapter_number)


def read_austen_chapters(work_name: str, chapter_range: tuple = (1, 5), contents: list = austen_contents) -> str:
    """
    Read specific chapters from a work by Jane Austen.
    """
    with open('data/jane_austen_complete.txt', 'r', encoding='utf-8') as file:
        full_text = file.read()
    work_text = extract_work(work_name, contents, full_text)
    del full_text  # Free memory
    chapter_texts = []
    for chapter_number in range(chapter_range[0], chapter_range[1] + 1):
        try:
            chapter_texts.append(extract_chapter(work_text, chapter_number))
        except ValueError as e:
            print(f"Error extracting chapter {chapter_number}: {e}")
    return ' '.join(chapter_texts) if chapter_texts else None


def read_austen_work(work_name: str, contents: list = austen_contents) -> str:
    """
    Read a specific work by Jane Austen and return its text.
    """
    with open('data/jane_austen_complete.txt', 'r', encoding='utf-8') as file:
        full_text = file.read()
    full_work = extract_work(work_name, contents, full_text)
    del full_text  # Free memory
    return full_work

if __name__ == "__main__": 
    # E.g. read the first chapter of PERSUASION
    try:
        chapter_texts = read_austen_chapters('PERSUASION', (1, 3))
        print(chapter_texts)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")