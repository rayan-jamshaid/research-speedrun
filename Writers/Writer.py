import os
from datetime import datetime
from abc import ABC, abstractmethod


class BaseWriter(ABC):
    """
    Abstract base class for writing model evaluation results.
    Follows SOLID principles by providing a common interface for all writers.
    """

    def __init__(self, output_dir="../results"):
        """
        Parameters
        ----------
        output_dir : str
            Directory where output files will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def write(self, results: dict, filename: str) -> str:
        """
        Write model evaluation results to a file.

        Parameters
        ----------
        results : dict
            Dictionary returned by the Models class.

        filename : str
            Name of the file (without extension).

        Returns
        -------
        str
            Path to the generated file.
        """
        pass


class Writer:
    """
    Facade class that provides backward compatibility with the original interface.
    Delegates to specific writer implementations.
    """

    def __init__(self, output_dir="../results"):
        """
        Parameters
        ----------
        output_dir : str
            Directory where markdown files will be saved.
        """
        self.output_dir = output_dir
        self._markdown_writer = MarkdownWriter(output_dir)
        self._html_writer = HTMLWriter(output_dir)
        self._csv_writer = CSVWriter(output_dir)

    def write_to_md(self, results: dict, filename: str) -> str:
        """
        Write model evaluation results to a Markdown (.md) file.

        Parameters
        ----------
        results : dict
            Dictionary returned by the Models class.

        filename : str
            Name of the markdown file (without .md extension).

        Returns
        -------
        str
            Path to the generated markdown file.
        """
        return self._markdown_writer.write(results, filename)

    def write_to_html(self, results: dict, filename: str) -> str:
        """
        Write model evaluation results to an HTML (.html) file.

        Parameters
        ----------
        results : dict
            Dictionary returned by the Models class.

        filename : str
            Name of the HTML file (without .html extension).

        Returns
        -------
        str
            Path to the generated HTML file.
        """
        return self._html_writer.write(results, filename)

    def write_to_csv(self, results: dict, filename: str) -> str:
        """
        Write model evaluation results to a single CSV file (results.csv).

        Each row = one model, with columns for all metrics.

        Parameters
        ----------
        results : dict
            Dictionary returned by the Models class.

        filename : str
            Name/identifier for the model (used as first column).

        Returns
        -------
        str
            Path to the generated CSV file.
        """
        return self._csv_writer.write(results, filename)


class MarkdownWriter(BaseWriter):
    """
    Writer for Markdown format output.
    """

    def write(self, results: dict, filename: str) -> str:
        """
        Write model evaluation results to a Markdown (.md) file.

        Parameters
        ----------
        results : dict
            Dictionary returned by the Models class.

        filename : str
            Name of the markdown file (without .md extension).

        Returns
        -------
        str
            Path to the generated markdown file.
        """

        filepath = os.path.join(self.output_dir, f"{filename}.md")

        with open(filepath, "w", encoding="utf-8") as f:

            f.write(f"# Model Evaluation Report\n\n")
            f.write(f"Generated: {datetime.now()}\n\n")

            ###############################################################
            # Model
            ###############################################################

            model_name = type(results["model"]).__name__

            f.write(f"## Model\n\n")
            f.write(f"**{model_name}**\n\n")

            ###############################################################
            # Validation Metrics
            ###############################################################

            f.write("## Validation Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|------:|\n")

            for metric, value in results["validation"]["metrics"].items():
                f.write(f"| {metric} | {value:.4f} |\n")

            f.write("\n")

            ###############################################################
            # Test Metrics
            ###############################################################

            f.write("## Test Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|------:|\n")

            for metric, value in results["test"]["metrics"].items():
                f.write(f"| {metric} | {value:.4f} |\n")

            f.write("\n")

            ###############################################################
            # Validation Images
            ###############################################################

            f.write("## Validation Images\n\n")

            for name, path in results["validation"]["images"].items():
                f.write(f"### {name.replace('_', ' ').title()}\n\n")
                f.write(f"![{name}]({path})\n\n")

            ###############################################################
            # Test Images
            ###############################################################

            f.write("## Test Images\n\n")

            for name, path in results["test"]["images"].items():
                f.write(f"### {name.replace('_', ' ').title()}\n\n")
                f.write(f"![{name}]({path})\n\n")

            ###############################################################
            # External Metrics
            ###############################################################

            if "external" in results:
                f.write("## External Metrics\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|------:|\n")

                for metric, value in results["external"]["metrics"].items():
                    f.write(f"| {metric} | {value:.4f} |\n")

                f.write("\n")

                ###############################################################
                # External Images
                ###############################################################

                f.write("## External Images\n\n")

                for name, path in results["external"]["images"].items():
                    f.write(f"### {name.replace('_', ' ').title()}\n\n")
                    f.write(f"![{name}]({path})\n\n")

        return filepath


class HTMLWriter(BaseWriter):
    """
    Writer for HTML format output.
    """

    def write(self, results: dict, filename: str) -> str:
        """
        Write model evaluation results to an HTML (.html) file.

        Parameters
        ----------
        results : dict
            Dictionary returned by the Models class.

        filename : str
            Name of the HTML file (without .html extension).

        Returns
        -------
        str
            Path to the generated HTML file.
        """
        filepath = os.path.join(self.output_dir, f"{filename}.html")

        with open(filepath, "w", encoding="utf-8") as f:

            f.write("<!DOCTYPE html>\n")
            f.write("<html>\n")
            f.write("<head>\n")
            f.write('<meta charset="utf-8">\n')
            f.write(f"<title>Model Evaluation Report</title>\n")
            f.write("<style>\n")
            f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
            f.write("h1, h2, h3 { color: #333; }\n")
            f.write("table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }\n")
            f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
            f.write("th { background-color: #4CAF50; color: white; }\n")
            f.write("tr:nth-child(even) { background-color: #f2f2f2; }\n")
            f.write("img { max-width: 600px; height: auto; }\n")
            f.write("</style>\n")
            f.write("</head>\n")
            f.write("<body>\n")

            f.write("<h1>Model Evaluation Report</h1>\n")
            f.write(f"<p>Generated: {datetime.now()}</p>\n")
            f.write("<hr>\n")

            ###############################################################
            # Model
            ###############################################################

            model_name = type(results["model"]).__name__

            f.write("<h2>Model</h2>\n")
            f.write(f"<p><strong>{model_name}</strong></p>\n")
            f.write("<hr>\n")

            ###############################################################
            # Validation Metrics
            ###############################################################

            f.write("<h2>Validation Metrics</h2>\n")
            f.write("<table>\n")
            f.write("<tr><th>Metric</th><th>Value</th></tr>\n")

            for metric, value in results["validation"]["metrics"].items():
                f.write(f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>\n")

            f.write("</table>\n")
            f.write("<hr>\n")

            ###############################################################
            # Test Metrics
            ###############################################################

            f.write("<h2>Test Metrics</h2>\n")
            f.write("<table>\n")
            f.write("<tr><th>Metric</th><th>Value</th></tr>\n")

            for metric, value in results["test"]["metrics"].items():
                f.write(f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>\n")

            f.write("</table>\n")
            f.write("<hr>\n")

            ###############################################################
            # Validation Images
            ###############################################################

            f.write("<h2>Validation Images</h2>\n")

            for name, path in results["validation"]["images"].items():
                f.write(f"<h3>{name.replace('_', ' ').title()}</h3>\n")
                f.write(f'<img src="{path}" alt="{name}"><br><br>\n')

            ###############################################################
            # Test Images
            ###############################################################

            f.write("<h2>Test Images</h2>\n")

            for name, path in results["test"]["images"].items():
                f.write(f"<h3>{name.replace('_', ' ').title()}</h3>\n")
                f.write(f'<img src="{path}" alt="{name}"><br><br>\n')

            ###############################################################
            # External Metrics
            ###############################################################

            if "external" in results:
                f.write("<h2>External Metrics</h2>\n")
                f.write("<table>\n")
                f.write("<tr><th>Metric</th><th>Value</th></tr>\n")

                for metric, value in results["external"]["metrics"].items():
                    f.write(f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>\n")

                f.write("</table>\n")
                f.write("<hr>\n")

                ###############################################################
                # External Images
                ###############################################################

                f.write("<h2>External Images</h2>\n")

                for name, path in results["external"]["images"].items():
                    f.write(f"<h3>{name.replace('_', ' ').title()}</h3>\n")
                    f.write(f'<img src="{path}" alt="{name}"><br><br>\n')

            f.write("</body>\n")
            f.write("</html>\n")

        return filepath


class CSVWriter(BaseWriter):
    """
    Writer for CSV format output.
    """

    def write(self, results: dict, filename: str) -> str:
        """
        Write model evaluation results to a single CSV file (results.csv).

        Each row = one model, with columns for all metrics.

        Parameters
        ----------
        results : dict
            Dictionary returned by the Models class.

        filename : str
            Name/identifier for the model (used as first column).

        Returns
        -------
        str
            Path to the generated CSV file.
        """
        import csv

        filepath = os.path.join(self.output_dir, "results.csv")

        # Define column order
        cm_labels = ["true_positive", "true_negative", "false_positive", "false_negative"]
        metric_names = ["accuracy", "precision", "recall", "f1", "f2_score", "specificity", "npv", "fpr", "fnr", "roc_auc", "pr_auc"]
        datasets = ["validation", "test"]
        if "external" in results:
            datasets.append("external")

        # Build row data
        row = [filename]  # First column is the model name

        for dataset_name in datasets:
            dataset_data = results[dataset_name]
            # Add metrics
            for metric in metric_names:
                value = dataset_data["metrics"].get(metric, None)
                row.append(value if value is not None else "")
            # Add confusion matrix values
            confusion_matrix = dataset_data.get("confusion_matrix", {})
            for label in cm_labels:
                value = confusion_matrix.get(label, 0)
                row.append(value)

        # Write to CSV
        write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
        try:
            with open(filepath, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # If file is empty, write header
                if write_header:
                    header = ["model_name"]
                    for dataset_name in datasets:
                        for metric in metric_names:
                            header.append(f"{dataset_name}_{metric}")
                        for label in cm_labels:
                            header.append(f"{dataset_name}_{label}")
                    writer.writerow(header)
                writer.writerow(row)
        except PermissionError:
            import random
            fallback_filepath = os.path.join(self.output_dir, f"results_fallback_{random.randint(1000, 9999)}.csv")
            print(f"[WARNING] Permission denied when writing to {filepath}. The file may be open in another application (like Excel). Writing results to: {fallback_filepath}")
            try:
                with open(fallback_filepath, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    header = ["model_name"]
                    for dataset_name in datasets:
                        for metric in metric_names:
                            header.append(f"{dataset_name}_{metric}")
                        for label in cm_labels:
                            header.append(f"{dataset_name}_{label}")
                    writer.writerow(header)
                    writer.writerow(row)
                return fallback_filepath
            except Exception as e:
                print(f"[ERROR] Failed to write fallback CSV file: {e}")
                return filepath

        return filepath
