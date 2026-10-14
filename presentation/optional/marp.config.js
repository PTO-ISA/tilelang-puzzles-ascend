// Marp-only build settings. The Beamer build (presentation/Makefile) does not
// read this file.
//
// `typographer` turns the `--` written in slides.md into proper en dashes, which
// is what pandoc does for the Beamer build; without it the two decks differ for
// a reason that has nothing to do with either tool's layout.
module.exports = {
  options: {
    markdown: {
      typographer: true,
      html: false,
    },
  },
};
