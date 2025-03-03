#include <CsLogHandle.h>

// The log handle
CsLogHandle::CsLogHandle(std::string font) {
    // For some reason relative path doesn't work
    // Lets just use the absolute path
    std::string dir = "assets/Fonts/" + font;
    fonts[0].loadFromFile(dir + "-Regular.ttf");
    fonts[1].loadFromFile(dir + "-Bold.ttf");
    fonts[2].loadFromFile(dir + "-Italic.ttf");
    fonts[3].loadFromFile(dir + "-BoldItalic.ttf");
}

// Add log
void CsLogHandle::addLog(std::string log, sf::Color color, short style) {
    CsLog cslog = {log, color, style};
    cslogs.push_back(cslog);
}
void CsLogHandle::addLog(std::string log, short style) {
    addLog(log, sf::Color::White, style);
}
// Clear log
void CsLogHandle::clear() {
    cslogs.clear();
}

// Draw log
void CsLogHandle::drawLog(sf::RenderWindow& window) {
    if (!displayLog) {
        cslogs.clear();
        return;
    };

    for (int i = 0; i < cslogs.size(); i++) {
        // Remove the /n at the end of the log
        // As it is unnecessary
        if (cslogs[i].log.back() == '\n')
            cslogs[i].log.pop_back();

        // Count the number of /n in the previous logs
        int count = 0;
        if (i != 0) count = std::count(
            cslogs[i - 1].log.begin(), cslogs[i - 1].log.end(), '\n'
        );

        sf::Text text;
        text.setString(cslogs[i].log);
        text.setFillColor(cslogs[i].color);
        text.setFont(fonts[cslogs[i].style]);
        // Sizing and positioning
        text.setCharacterSize(fontSize);
        text.setPosition(marginL, marginT + fontSize*1.5 * (i + count));

        window.draw(text);
    }

    cslogs.clear();
}