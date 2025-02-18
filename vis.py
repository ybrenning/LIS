import json
from spacy import displacy
from bs4 import BeautifulSoup


def generate_docs(filename, display=False):
    """
    Generate visualization docs from a file of triples.

    For example, the following sentence:

        Comment	ADV	shift	3
        t'	PRON	shift	3
        appelles	VERB	left-arc left-arc right-arc	0
        -tu	ADV	right-arc	3
        ,	PUNCT	shift	6
        prophétesse	VERB	left-arc reduce right-arc	3
        de	ADP	shift	8
        bonheur	NOUN	left-arc right-arc	6
        ?	PUNCT	reduce reduce right-arc reduce	3

    Generates the following doc:

    {
        "words": [
            {
                "text": "Comment",
                "tag": "ADV"
            },
            {
                "text": "t'",
                "tag": "PRON"
            },
            {
                "text": "appelles",
                "tag": "VERB"
            },
            ...
        ],
        "arcs": [
            {
                "start": 0,
                "end": 2,
                "label": "shift",
                "dir": "left"
            },
            {
                "start": 1,
                "end": 2,
                "label": "shift",
                "dir": "left"
            },
            ...
        ]
    }
    """
    docs = []

    with open(filename, "r") as f:
        words = []
        arcs = []
        i = 0
        for line in f.readlines():
            cols = line.strip().split("\t")

            if len(cols) <= 1:
                assert words and arcs
                doc = {"words": words, "arcs": arcs}

                if display:
                    print(json.dumps(doc, indent=4))
                docs.append(doc)

                i = 0
                words = []
                arcs = []
                continue

            words.append(
                {"text": cols[0], "tag": cols[1]}
            )

            # Ignore root node as it is not displayed
            if int(cols[3]) != 0:
                start = int(cols[3]) - 1
                end = i
                if start > end:
                    dir = "left"
                    end = start
                    start = i
                else:
                    dir = "right"

                arcs.append(
                    {
                        "start": start,
                        "end": end,
                        "label": cols[2],
                        "dir": dir
                    }
                )

            i += 1

    return docs


def render_html(docs):
    """
    Render an HTML page containing dependency relations
    and a short description based on a provided list of docs.
    """
    html = displacy.render(
        docs, style="dep", manual=True, page=True, options={"distance": 125},
    )
    soup = BeautifulSoup(html, "html.parser")

    new_title_text = "Visualization of action lists with relations"
    if soup.title:
        soup.title.string = new_title_text
    else:
        new_title = soup.new_tag("title")
        new_title.string = new_title_text
        if soup.head:
            soup.head.insert(0, new_title)
        else:
            new_head = soup.new_tag("head")
            new_head.insert(0, new_title)
            soup.html.insert(0, new_head)
    new_heading = soup.new_tag("h1")
    new_heading.string = new_title_text
    if soup.body:
        soup.body.insert(0, new_heading)

    new_tag = soup.new_tag("p")
    new_tag.string = """
    We can visualize dependency relations of 50 short sentences from the Dumas
    corpus along with each words' "action lists" and POS tags;
    the arrow labels contain the "action list" of the dependant.
    """

    new_tag.append(soup.new_tag("br"))
    new_tag.append(soup.new_tag("br"))
    new_tag.append(
        """
        Note that displaCy doesn't explicitly show the root relation.
        This means that the root of the sentence in this type of visualization
        is always the word that has no head within the sentence.
        """
    )

    new_tag.append(soup.new_tag("br"))
    new_tag.append(soup.new_tag("br"))
    new_tag.append(
        """
        The relations and action lists were predicted by IncPar and are not
        necessarily correct. For example, the results seem to have predicted
        multiple root words in some cases.
        """
    )

    if soup.body:
        soup.body.insert(1, new_tag)

    style_tag = soup.new_tag("style")
    style_tag.string = """
        h1 { font-size: 40px; }
        p { font-size: 25px; }
    """
    if soup.head:
        soup.head.append(style_tag)

    return str(soup)


def main():
    input = "gutenberg_dumas.vis"
    output = "vis.html"

    docs = generate_docs(input)
    html = render_html(docs)

    with open(output, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
