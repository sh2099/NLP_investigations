import re

# Find start of each work in the full text
def find_start(work_name, text):
    title = re.findall(f'{work_name.upper()}', text)
    if title:
        start_index = text.index(title[0])
        # Now search again for the second occurrence of the work name
        second_idx = text.index(title[0], start_index + len(title[0]))
        return second_idx
    else:
        raise ValueError(f"Work '{work_name}' not found in the text.")
    
def find_end(work_name, contents, text):
    next_work = contents[contents.index(work_name) + 1] if contents.index(work_name) + 1 < len(contents) else None
    if next_work:
        end_index = find_start(next_work, text)
        return end_index
    else:
        return len(text)  # If it's the last work, return the end of the text

def extract_work(work_name, contents, text):
    start_index = find_start(work_name, text)
    end_index = find_end(work_name, contents, text)
    return text[start_index:end_index].strip()

def extract_chapter(work, chapter_number=1):
    # Split the work into chapters
    chapters = re.split(r'\n\s*Chapter \d+\s*\n', work)
    if chapter_number <= len(chapters):
        # Chapter 0 is intro/preface which we skip
        return chapters[chapter_number].strip()
    else:
        raise ValueError(f"Chapter {chapter_number} does not exist in the work.")


def read_austen_chapter(work_name, chapter_number=1):
    """
    Read a specific work by Jane Austen and return the specified chapter.
    
    :param work_name: Name of the work (e.g., 'PERSUASION', 'NORTHANGER ABBEY', etc.)
    :param chapter_number: Chapter number to extract (default is 1)
    :return: The text of the specified chapter
    """
    with open('data/jane_austen_complete.txt', 'r', encoding='utf-8') as file:
        full_text = file.read()
    contents = [
        'PERSUASION',
        'NORTHANGER ABBEY',
        'MANSFIELD PARK',
        'EMMA',
        'LADY SUSAN',
        'LOVE AND FRIENDSHIP',
        'PRIDE AND PREJUDICE AND OTHER EARLY WORKS',
        'SENSE AND SENSIBILITY',
    ]
    work_text = extract_work(work_name, contents, full_text)
    del full_text  # Free memory
    return extract_chapter(work_text, chapter_number)


if __name__ == "__main__": 
    # Example usage
    try:
        chapter_text = read_austen_chapter('PERSUASION', 24)
        print(chapter_text)
    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")