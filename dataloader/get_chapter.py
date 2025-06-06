import re

with open('jane_austen_complete.txt', 'r', encoding='utf-8') as file:
    full_texts = file.read()


# First get list of works from the file contents information
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


# Find start of each work in the full text
def find_start(work_name, text=full_texts):
    title = re.findall(f'{work_name.upper()}', text)
    if title:
        start_index = text.index(title[0])
        # Now search again for the second occurrence of the work name
        second_idx = text.index(title[0], start_index + len(title[0]))
        return second_idx
    else:
        raise ValueError(f"Work '{work_name}' not found in the text.")
    
def find_end(work_name, text=full_texts):
    next_work = contents[contents.index(work_name) + 1] if contents.index(work_name) + 1 < len(contents) else None
    if next_work:
        end_index = find_start(next_work, text)
        return end_index
    else:
        return len(text)  # If it's the last work, return the end of the text

def extract_work(work_name, text=full_texts):
    start_index = find_start(work_name, text)
    end_index = find_end(work_name, text)
    return text[start_index:end_index].strip()

def extract_chapter(work, chapter_number=1):
    # Split the work into chapters
    chapters = re.split(r'\n\s*Chapter \d+\s*\n', work)
    if chapter_number <= len(chapters):
        # Chapter 0 is intro/preface which we skip
        return chapters[chapter_number].strip()
    else:
        raise ValueError(f"Chapter {chapter_number} does not exist in the work.")
    
    
persuasion = extract_work('PERSUASION')
persuasion_ch1 = extract_chapter(persuasion, 1)
print(persuasion_ch1[:100])

