from typing import List

class RecursiveChunker:

    """
    Recursive Text Chunk for preserving semantic boundaries.
    Split in priority order:
    1. Paragraphs (\\n\\n)
    2. Line breaks (\\n)
    3. Sentence boundaries (.)
    4. Spaces
    """

    def __init__(self, chunk_size:int=1024, overlap:int=128):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n", "\n", ". ", " "]


    # split text and return a list of valid chunks
    def split(self, text:str) -> List[str]:
        """
        Split text into a list of chunks
        """
        text = text.strip() # Remove leading and trailing whitespace

        if not text: # If the text is empty
            return []

        # Estimate token count (roughly 1 token ≈ 4 characters)
        # In production, use tiktoken for accurate counting; simplified here

        chunks = self._split_recursive(text,self.separators)

        result = []
        for c in chunks:
            if c.strip(): # If c is valid text (not empty)
                result.append(c.strip())
        return result

    def _split_recursive(self, text:str, separators:List[str]) -> List[str]:

        """
        Recursively splits text into smaller chunks using multiple separators
        (paragraph → line → sentence → space).

        Core idea:
        - Try to split text using the current separator.
        - Build chunks until reaching the size limit.
        - If a chunk is too large, recursively use a finer separator.
        - If needed, force split at character level.
        - Add overlap between chunks to preserve context.

        Example:
            Input:
                "Hello world. This is a test."

            Process:
                1. Try ". " → ["Hello world", "This is a test"]
                2. If a part is still too large → split by space
                3. If still too large → split character by character

            Output:
                ["Hello world", "This is a test"]
        """

        # If text is too short, return
        if len(text) // 4 <= self.chunk_size:
            return[text]

        # get separators
        separators = separators[0] if separators else " "

        # Split the text into parts using specified separator. e.g. ['a','b','c']
        parts = text.split(separators)

        chunks = []
        current_chunk = ""

        for part in parts:
            test_chunk = current_chunk + separators + part if current_chunk else part

            if len(test_chunk) // 4 > self.chunk_size:
                # if chunk is full
                if current_chunk:
                    chunks.append(current_chunk)

                    #overlap: carry over the end of the previous chunk to the next
                    overlap_text = current_chunk[-(self.overlap*4):]
                    current_chunk = overlap_text + separators + part if overlap_text else part

                else:
                    # Single part is too long; use finer separators recursively
                    if len(separators) > 1:
                        sub_chunks = self._split_recursive(part,separators[1:])
                        chunks.extend(sub_chunks)
                    else:
                        # 已经是最细粒度，强制截断
                        chunks.append(part[:self.chunk_size * 4])
                    current_chunk = ""

            else:
                current_chunk = test_chunk

        if current_chunk:
            chunks.append(current_chunk)

        return chunks






