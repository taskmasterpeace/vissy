import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, Scrollbar, Text
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from collections import Counter
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import datetime
import re

class Vissy:
    def __init__(self, master):
        self.master = master
        self.master.title("Vissy - Interactive Transcript Analysis Tool")
        self.master.geometry("1200x800")

        self.transcripts_data = []
        self.word_counts = Counter()
        self.file_paths = {}  # Store full file paths
        self.create_widgets()

        # Ensure NLTK and TextBlob corpora are downloaded
        self.download_required_data()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel for controls
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # File selection
        ttk.Label(left_panel, text="Select Transcripts:").pack(anchor=tk.W)
        self.file_listbox = tk.Listbox(left_panel, selectmode=tk.EXTENDED, width=50, height=10)
        self.file_listbox.pack(fill=tk.X, pady=5)
        ttk.Button(left_panel, text="Browse", command=self.select_files).pack(anchor=tk.W)
        ttk.Button(left_panel, text="Select All", command=self.select_all_files).pack(anchor=tk.W)
        ttk.Button(left_panel, text="Analyze Selected", command=self.analyze_transcripts).pack(anchor=tk.W, pady=5)

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(left_panel, textvariable=self.status_var).pack(anchor=tk.W, pady=5)

        # Top percentage selection
        ttk.Label(left_panel, text="Top Words:").pack(anchor=tk.W, pady=(10,0))
        self.top_mode = tk.StringVar(value="number")
        self.top_number = tk.StringVar(value="10")
        self.top_mode_dropdown = ttk.Combobox(left_panel, textvariable=self.top_mode, values=["percentage", "number"], state="readonly")
        self.top_mode_dropdown.pack(anchor=tk.W)
        ttk.Entry(left_panel, textvariable=self.top_number, width=5).pack(anchor=tk.W)
        ttk.Button(left_panel, text="Update Visualizations", command=self.update_visualizations).pack(anchor=tk.W, pady=5)

        # Exclude words input
        ttk.Label(left_panel, text="Exclude words (comma-separated):").pack(anchor=tk.W, pady=(10,0))
        self.exclude_words_var = tk.StringVar()
        self.exclude_words_entry = ttk.Entry(left_panel, textvariable=self.exclude_words_var, width=40)
        self.exclude_words_entry.pack(anchor=tk.W)

        # Search frame
        ttk.Label(left_panel, text="Search word:").pack(anchor=tk.W, pady=(10,0))
        self.search_var = tk.StringVar()
        ttk.Entry(left_panel, textvariable=self.search_var, width=20).pack(anchor=tk.W)
        ttk.Button(left_panel, text="Search", command=self.search_word).pack(anchor=tk.W, pady=5)

        # Filter by word length
        ttk.Label(left_panel, text="Min word length:").pack(anchor=tk.W, pady=(10,0))
        self.min_word_length = tk.StringVar(value="1")
        ttk.Entry(left_panel, textvariable=self.min_word_length, width=5).pack(anchor=tk.W)

        # Right panel for visualizations
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas for scrollable notebook
        canvas = tk.Canvas(right_panel)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar for canvas
        scrollbar = ttk.Scrollbar(right_panel, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)

        # Notebook within a frame in the canvas
        self.notebook_frame = ttk.Frame(canvas)
        self.notebook_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.notebook_frame, anchor="nw")

        # Notebook for visualizations
        self.notebook = ttk.Notebook(self.notebook_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tabs for visualizations
        self.wordcloud_tab = ttk.Frame(self.notebook)
        self.barchart_tab = ttk.Frame(self.notebook)
        self.timeseries_tab = ttk.Frame(self.notebook)
        self.details_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.wordcloud_tab, text='Word Cloud')
        self.notebook.add(self.barchart_tab, text='Top Words')
        self.notebook.add(self.timeseries_tab, text='Word Frequency Over Time')
        self.notebook.add(self.details_tab, text='Word Details')

        self.bind_shift_click()

    def bind_shift_click(self):
        self.file_listbox.bind('<Shift-Button-1>', self.on_shift_click)

    def on_shift_click(self, event):
        widget = event.widget
        selection = widget.curselection()
        if selection:
            last_selected = selection[-1]
            widget.selection_anchor(last_selected)
            widget.selection_set(widget.index('@%d,%d' % (event.x, event.y)))
        else:
            widget.selection_anchor(0)
            widget.selection_set(widget.index('@%d,%d' % (event.x, event.y)))

    def select_all_files(self):
        self.file_listbox.select_set(0, tk.END)

    def select_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Text files", "*.txt")])
        for file in files:
            if self.is_transcript_file(file):
                filename = os.path.basename(file)
                if filename not in self.file_paths:
                    self.file_listbox.insert(tk.END, filename)
                    self.file_paths[filename] = file
        self.file_listbox.select_set(0, tk.END)
        self.status_var.set(f"Selected {len(files)} transcript files")

    def is_transcript_file(self, filename):
        base_filename = os.path.basename(filename)
        pattern = r'^\d{6}.*tra ?nscri ?pt\.txt$'
        if re.search(pattern, base_filename, re.IGNORECASE):
            return True
        if re.search(r'^\d{6}_\d{4}.*tra ?nscri ?pt\.txt$', base_filename, re.IGNORECASE):
            return True
        return False

    def download_required_data(self):
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            messagebox.showerror("Data Download Error", f"Error downloading required NLTK data: {str(e)}")

        try:
            from textblob.download_corpora import download_all
            download_all()
        except Exception as e:
            messagebox.showerror("Data Download Error", f"Error downloading TextBlob corpora: {str(e)}")

    def analyze_transcripts(self):
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Files Selected", "Please select transcript files to analyze.")
            return

        self.status_var.set("Analyzing transcripts...")
        self.master.update()

        self.transcripts_data = []
        self.word_counts = Counter()

        for index in selected_indices:
            filename = self.file_listbox.get(index)
            file_path = self.file_paths[filename]
            date = self.extract_date_from_filename(filename)
            
            try:
                # Try different encodings
                encodings = ['utf-8', 'iso-8859-1', 'windows-1252']
                content = None
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as file:
                            content = file.read()
                        break
                    except UnicodeDecodeError:
                        continue

                if content is None:
                    raise UnicodeDecodeError(f"Unable to decode file {file_path} with any of the attempted encodings.")

                if not content.strip():
                    print(f"Warning: File {file_path} is empty.")
                    continue

                words = self.preprocess_text(content)
                if not words:
                    print(f"Warning: No valid words found in file {file_path} after preprocessing.")
                    continue

                blob = TextBlob(content)
                sentiment = blob.sentiment
                phrases = blob.noun_phrases

                self.word_counts.update(words)
                self.transcripts_data.append({
                    'date': date,
                    'content': content,
                    'words': words,
                    'sentiment': sentiment,
                    'phrases': phrases
                })
                print(f"Processed file: {file_path}")
            except ValueError as ve:
                print(f"Error processing file {file_path}: {ve}")
                messagebox.showerror("File Processing Error", f"Error processing file {file_path}: {ve}")
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                messagebox.showerror("File Processing Error", f"Error processing file {file_path}: {str(e)}")

        if not self.transcripts_data:
            messagebox.showerror("Analysis Error", "No valid transcript data found in selected files.")
            self.status_var.set("Analysis failed")
            return

        self.transcripts_data.sort(key=lambda x: x['date'])
        self.status_var.set(f"Analysis complete. Processed {len(self.transcripts_data)} files.")
        self.update_visualizations()

    def preprocess_text(self, text):
        stop_words = set(stopwords.words('english'))
        exclude_words = set(self.exclude_words_var.get().split(','))
        words = word_tokenize(text.lower())
        min_length = int(self.min_word_length.get())
        return [word for word in words if word.isalnum() and word not in stop_words and word not in exclude_words and len(word) >= min_length]

    def update_visualizations(self):
        if not self.transcripts_data:
            messagebox.showwarning("No Data", "No transcript data available. Please analyze files first.")
            return
        if not self.word_counts:
            messagebox.showwarning("No Words", "No words found in the transcripts. Cannot create visualizations.")
            return
        self.create_wordcloud()
        self.create_top_words_barchart()
        self.create_word_frequency_timeseries()

    def create_wordcloud(self):
        for widget in self.wordcloud_tab.winfo_children():
            widget.destroy()

        if not self.word_counts:
            ttk.Label(self.wordcloud_tab, text="No words to display in word cloud.").pack()
            return

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(self.word_counts)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud of Transcripts')

        canvas = FigureCanvasTkAgg(fig, master=self.wordcloud_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.add_toolbar(canvas, self.wordcloud_tab)

    def create_top_words_barchart(self):
        for widget in self.barchart_tab.winfo_children():
            widget.destroy()

        if not self.word_counts:
            ttk.Label(self.barchart_tab, text="No words to display in bar chart.").pack()
            return

        filter_mode = self.top_mode.get()
        top_value = int(self.top_number.get())
        if filter_mode == "percentage":
            top_n = max(1, int(len(self.word_counts) * top_value / 100))
        elif filter_mode == "number":
            top_n = top_value
        top_words = dict(self.word_counts.most_common(top_n))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(top_words.keys(), top_words.values())
        ax.set_title(f'Top {top_value}% Words in Transcripts' if filter_mode == "percentage" else f'Top {top_n} Words in Transcripts')
        ax.set_xlabel('Words')
        ax.set_ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')

        canvas = FigureCanvasTkAgg(fig, master=self.barchart_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.add_toolbar(canvas, self.barchart_tab)

    def create_word_frequency_timeseries(self):
        for widget in self.timeseries_tab.winfo_children():
            widget.destroy()

        if not self.transcripts_data or not self.word_counts:
            ttk.Label(self.timeseries_tab, text="No data to display in time series.").pack()
            return

        top_words = dict(self.word_counts.most_common(10))
        dates = [transcript['date'] for transcript in self.transcripts_data]
        word_freq_over_time = {word: [sum(1 for w in transcript['words'] if w == word) for transcript in self.transcripts_data] for word in top_words}

        fig, ax = plt.subplots(figsize=(10, 5))
        self.lines = {}
        for word, freq in word_freq_over_time.items():
            line, = ax.plot(dates, freq, label=word, marker='o')
            self.lines[word] = line

        ax.set_title('Top 10 Words Frequency Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Frequency')
        ax.legend()
        plt.xticks(rotation=45, ha='right')

        canvas = FigureCanvasTkAgg(fig, master=self.timeseries_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.add_toolbar(canvas, self.timeseries_tab)

        # Adding checkboxes for selecting words
        self.add_word_selection_checkboxes()

    def add_word_selection_checkboxes(self):
        frame = ttk.Frame(self.timeseries_tab)
        frame.pack(side=tk.BOTTOM, fill=tk.X)

        for word in self.lines.keys():
            var = tk.BooleanVar(value=True)
            chk = tk.Checkbutton(frame, text=word, variable=var, command=lambda w=word, v=var: self.toggle_word_visibility(w, v))
            chk.pack(side=tk.LEFT)

    def toggle_word_visibility(self, word, var):
        line = self.lines[word]
        line.set_visible(var.get())
        self.notebook.update_idletasks()

    def search_word(self):
        search_term = self.search_var.get().lower()
        if not search_term or not self.transcripts_data:
            messagebox.showwarning("Invalid Search", "Please enter a search term and ensure transcripts are analyzed.")
            return

        search_terms = [term.strip() for term in search_term.split(',')]
        search_results = []
        for transcript in self.transcripts_data:
            matches = {term: transcript['content'].lower().count(term) for term in search_terms}
            if any(matches.values()):
                context = self.get_context(transcript['content'], search_terms)
                search_results.append((transcript['date'], context, matches))

        self.display_search_results(search_term, search_results)

    def get_context(self, content, search_terms, window=5):
        sentences = nltk.sent_tokenize(content)
        context = []
        for term in search_terms:
            for i, sentence in enumerate(sentences):
                if term in sentence:
                    start = max(0, i - window)
                    end = min(len(sentences), i + window + 1)
                    context.extend(sentences[start:end])
        return context

    def display_search_results(self, search_term, search_results):
        for widget in self.details_tab.winfo_children():
            widget.destroy()

        if not search_results:
            ttk.Label(self.details_tab, text="No results found for the search term.").pack()
            return

        frame = ttk.Frame(self.details_tab)
        frame.pack(fill=tk.BOTH, expand=True)

        text_widget = tk.Text(frame, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True)

        transcript_vars = []
        for date, context, matches in search_results:
            var = tk.BooleanVar(value=True)
            transcript_vars.append((var, context))
            chk = tk.Checkbutton(frame, text=f"{date} (Matches: {matches})", variable=var)
            chk.pack(anchor=tk.W)
            text_widget.insert(tk.END, f"{date}:\n")
            for sentence in context:
                highlighted_context = sentence
                for term in matches.keys():
                    highlighted_context = highlighted_context.replace(term, f"<<{term}>>")
                text_widget.insert(tk.END, highlighted_context + " ")
            text_widget.insert(tk.END, "\n\n")

        ttk.Button(frame, text="Copy Selected Transcripts", command=lambda: self.copy_selected_transcripts(transcript_vars)).pack(anchor=tk.W)

    def copy_selected_transcripts(self, transcript_vars):
        selected_content = "\n\n".join(" ".join(context) for var, context in transcript_vars if var.get())
        self.master.clipboard_clear()
        self.master.clipboard_append(selected_content)
        messagebox.showinfo("Copy Successful", "Selected transcripts copied to clipboard.")

    def add_toolbar(self, canvas, parent):
        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def extract_date_from_filename(self, filename):
        match = re.search(r'^(\d{2})(\d{2})(\d{2})', os.path.basename(filename))
        if match:
            year, month, day = match.groups()
            year = f"20{year}"  # Assuming all years are in the 21st century
            return datetime.datetime(int(year), int(month), int(day))
        return None

if __name__ == "__main__":
    root = tk.Tk()
    app = Vissy(root)
    root.mainloop()
