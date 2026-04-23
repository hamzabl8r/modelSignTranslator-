import os

class SentenceManager:
    def __init__(self, stability_threshold=20):
        self.sentence = []
        self.current_prediction = ""
        self.last_confirmed_word = ""
        self.frame_counter = 0
        self.stability_threshold = stability_threshold

    def process_prediction(self, predicted_word):
        """Processes each sequence prediction to build a sentence of words."""
        if predicted_word:
            # Check if the word is held steady over multiple frames/sequences
            if predicted_word == self.current_prediction:
                self.frame_counter += 1
            else:
                self.frame_counter = 0
                self.current_prediction = predicted_word

            # If the word is stable enough, add it to the list
            if self.frame_counter == self.stability_threshold:
                # Avoid adding the exact same word twice in a row immediately
                if predicted_word != self.last_confirmed_word:
                    self.sentence.append(predicted_word)
                    self.last_confirmed_word = predicted_word
                
                # Reset counter to wait for the next distinct sign
                self.frame_counter = 0
        else:
            # If hands leave the frame, reset 'last_confirmed' 
            # This allows the user to repeat the same word after a pause
            self.last_confirmed_word = ""
            self.current_prediction = ""
            self.frame_counter = 0

        # Return the sentence joined by spaces
        return " ".join(self.sentence)

    def save_to_file(self, filename="translations.txt"):
        """Saves current sentence and clears it."""
        full_text = " ".join(self.sentence)
        if full_text:
            with open(filename, "a") as f:
                f.write(full_text + "\n")
            self.sentence = []
            self.last_confirmed_word = "" # Reset for the next session
            return True
        return False