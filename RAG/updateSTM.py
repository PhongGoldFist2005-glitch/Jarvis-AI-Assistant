from langchain_text_splitters import RecursiveCharacterTextSplitter
from State import State

def update_STM(state: State):
	summary = state.get("summary")
	stm = state.get("stm")
	if not summary or stm is None:
		return {"stm": stm}

	splitter = RecursiveCharacterTextSplitter(
		chunk_size=240,
		chunk_overlap=40,
	)
	chunks = splitter.split_text(summary)

	for chunk in chunks:
		text = chunk.strip()
		if text:
			stm.insert_into_window(text)

	return {"stm": stm}
    