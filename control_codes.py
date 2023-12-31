START_C_CODES = {
    "admin": ":förvaltning:",
    "ads": ":anons:",
    "blogs": ":blogg:",
    "blogs/economy": ":blogg::ekonomi:",
    "blogs/sport": ":blogg::idrott:",
    "blogs/tech": ":blogg::teknologi:",
    "debate": ":debatt:",
    "forum": ":forum:",
    "forum/economy": ":forum::ekonomi:",
    "forum/law": ":forum::juridik:",
    "forum/sport": ":forum::idrott:",
    "forum/tech": ":forum::teknologi:",
    "forum/travel": ":forum::resor:",
    "info": ":info:",
    "info/business": ":info::affar:",
    "info/lifestyle": ":info::livstil:",
    "info/medical": ":info::med:",
    "info/travel": ":info::resor:",
    "news": ":nyheter:",
    "news/culture": ":nyheter::kultur:",
    "news/economy": ":nyheter::ekonomi:",
    "news/fashion": ":nyheter::mode:",
    "news/food": ":nyheter::mat:",
    "news/lifestyle": ":nyheter::livstil:",
    "news/opinion": ":nyheter::asikt:",
    "news/politics": ":nyheter::politik:",
    "news/pressrelease": ":nyheter::pressmeddelande:",
    "news/science": ":nyheter::vetenskap:",
    "news/sport": ":nyheter:idrott:",
    "news/sustainability": ":nyheter::hallbarhet:",
    "news/tech": ":nyheter::teknologi:",
    "news/travel": ":nyheter::resor:",
    "news/weather": ":nyheter::vader:",
    "review": ":recension:",
    "simple": ":lattlast:",
    "wiki": ":wiki:",
    "lit": ":litteratur:",
    "title": ":rubrik:"
}

END_C_CODES = {k: "{}$".format(v) for k, v in START_C_CODES.items()}

ADD_START_C_CODES = {    
    "mcq": ":mcq:"
}
ADD_END_C_CODES = {k: "{}$".format(v) for k, v in ADD_START_C_CODES.items()}

START_C_CODES.update(ADD_START_C_CODES)
END_C_CODES.update(ADD_END_C_CODES)
