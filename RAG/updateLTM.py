from langchain_text_splitters import RecursiveCharacterTextSplitter
from State import State


def update_LTM(state: State):
	summary = state.get("summary")
	ltm = state.get("ltm")
	if not summary or ltm is None:
		return {"ltm": ltm}

	splitter = RecursiveCharacterTextSplitter(
		chunk_size=240,
		chunk_overlap=40,
	)
	chunks = splitter.split_text(summary)

	for chunk in chunks:
		text = chunk.strip()
		if text:
			ltm.insert(text)

	return {"ltm": ltm}
