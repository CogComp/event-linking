import json, pickle


def GenerateBlinkData(infile_path, outfile_path):
	text_file = open('data/wikipedia/wiki/title_text.json', 'r')
	title_text = json.load(text_file)
	print('Text loaded.')

	id2title_file = open('data/wikipedia/wiki/enwiki-20200301.id2t.pkl', 'rb')
	id2title = pickle.load(id2title_file)
	print('ID to title loaded.')

	t2hyperlinks_file = open('data/wikipedia/wiki/t2hyperlinks.json', 'r')
	t2h = json.load(t2hyperlinks_file)

	infile = open(infile_path, 'r')
	clusters = json.load(infile)

	query_id = 0
	json_list = []

	for event in clusters:
		for mention in clusters[event]:
			event_form = event
			if event[0] == "'" and event[-1] == "'":
				event_form = event[1:-1]
			page = title_text[mention['page']]
			entities = []
			for idx, entity in enumerate(mention['entities']):
				form = title_text[mention['page']][entity['start']:entity['end']].lower()
				entity['form'] = form
				entities.append(entity)

			hyperlinks = []
			for idx, title in enumerate(t2h[id2title[1][event_form]]):
				if idx > 10:
					break
				link = t2h[id2title[1][event_form]][title]
				if link['end'] > 2000:
					continue
				hyperlinks.append((title, title_text[event_form.replace("_", ' ')][link['start']: link['end']]))

			json_list.append({
				"context_left": page[0: mention['start']],
				"mention": page[mention['start']: mention['end']],
				"context_right": page[mention['end']:],
				"label": title_text[event_form.replace("_", ' ')][0:2000],
				"label_title": event_form,
				"label_id": id2title[1][event_form],
				"hyperlinks": hyperlinks,
				"entities": entities
			})
			query_id += 1

	print(query_id)

	with open(outfile_path, 'w') as outfile:
		for entry in json_list:
			json.dump(entry, outfile)
			outfile.write('\n')


if __name__ == '__main__':
	GenerateBlinkData('data/wikipedia/preprocessed/wiki_valid.json', 'out/valid.jsonl')
	GenerateBlinkData('data/wikipedia/preprocessed/wiki_test.json', 'out/test.jsonl')
	GenerateBlinkData('data/wikipedia/preprocessed/wiki_train.json', 'out/train.jsonl')