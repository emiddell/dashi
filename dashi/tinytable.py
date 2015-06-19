
from jinja2 import Template
from collections import defaultdict




tmpl_html_h = Template("""
<html>
<head>
<style type="text/css">
 body { font-family: Arial; }
 table { border: solid 1px; }
 table th { background: #DDD; text-align: center;  }
 table tr { text-align: center;  }
</style>
</head>
<body>
<table>
 <tr>
   <th> </th>
   {% for y_label in y_labels %} <th> {{y_label }} </th> {% endfor %}
 </tr>
 {% for x_label in x_labels %}
 <tr>
   <td> {{x_label}} </td>
   {% for y_label in y_labels %} <td> {{ table_data[x_label][y_label] }} </td> {% endfor %}
 </tr>
 {% endfor %}
</table>
</body>
</html>
""")
    
tmpl_html_v = Template("""
<html>
<head>
<style type="text/css">
 body { font-family: Arial; }
 table { border: solid 1px; }
 table th { background: #DDD; text-align: center;  }
 table tr { text-align: center;  }
</style>
</head>
<body>
<table>
 <tr>
   <th> </th>
   {% for x_label in x_labels %} <th> {{x_label }} </th> {% endfor %}
 </tr>
 {% for y_label in y_labels %}
 <tr>
   <td> {{y_label}} </td>
   {% for x_label in x_labels %} <td> {{ table_data[x_label][y_label] }} </td> {% endfor %}
 </tr>
 {% endfor %}
</table>
</body>
</html>
""")
    
tmpl_wiki_h = Template("""{| class="wikitable"
|
{% for y_label in y_labels %}| align="center" | {{y_label }} 
{% endfor %}
|-
{% for x_label in x_labels %}| align="right" | {{x_label}}
{% for y_label in y_labels %}| align="right" | {{ table_data[x_label][y_label] }} 
{% endfor %}|-
{% endfor %}
|}""")
    
tmpl_wiki_v = Template("""{| class="wikitable"
|
{% for x_label in x_labels %}| align="center" | {{x_label }} 
{% endfor %}
|-
{% for y_label in y_labels %}| align="right" | {{y_label}}
{% for x_label in x_labels %}| align="right" | {{ table_data[x_label][y_label] }} 
{% endfor %}|-
{% endfor %}
|}""")
    
tmpl_latex_h = Template(r""" \begin{tabular}{ {% for y_label in range(y_labels|length + 1) %} c {% if y_labels|length + 1 - loop.index > 0 %} | {% endif %} {% endfor %}  }
& {% for y_label in y_labels %} {{y_label}} {% if y_labels|length - loop.index > 0 %} & {% else %} \\ \hline \hline {% endif %} {% endfor %} 
{% for x_label in x_labels %} {{x_label}} & {% for y_label in y_labels %} {{ table_data[x_label][y_label] }} {% if x_labels|length - loop.index > 0 %} & {% endif %} {% endfor %} {% if y_labels|length - loop.index > 0 %} \\ \hline {% endif %}  
{% endfor %}
\end{tabular}""")
    
tmpl_latex_v = Template(r""" \begin{tabular}{ {% for x_label in range(x_labels|length + 1) %} c {% if x_labels|length + 1 - loop.index > 0 %} | {% endif %} {% endfor %}  }
& {% for x_label in x_labels %} {{x_label}} {% if x_labels|length - loop.index > 0 %} & {% else %} \\ \hline \hline {% endif %} {% endfor %} 
{% for y_label in y_labels %} {{y_label}} & {% for x_label in x_labels %} {{ table_data[x_label][y_label] }} {% if y_labels|length - loop.index > 0 %} & {% endif %} {% endfor %} {% if x_labels|length - loop.index > 0 %} \\ \hline {% endif %}  
{% endfor %}
\end{tabular}""")

tmpl_latex_pretty_h = Template(r"""\begin{table}
\begin{tabularx}{1.\textwidth}{@{}{% for y_label in range(y_labels|length + 1) %}X{% if y_labels|length + 1 - loop.index > 0 %}{% endif %}{% endfor %}@{}}
\toprule
& {% for y_label in y_labels %} {{y_label}} {% if y_labels|length - loop.index > 0 %} & {% else %} \\ {% endif %} {% endfor %} 
\midrule
{% for x_label in x_labels %} {{x_label}} & {% for y_label in y_labels %} {{ table_data[x_label][y_label] }} {% if x_labels|length - loop.index > 0 %} & {% endif %} {% endfor %} {% if y_labels|length - loop.index > 0 %} \\ {% endif %}  
{% endfor %}
\bottomrule
\end{tabularx}
\caption{\label{tab:}}
\end{table}""")
    
tmpl_latex_pretty_v = Template(r"""\begin{table}
\begin{tabularx}{1.\textwidth}{@{}{% for x_label in range(x_labels|length + 1) %}X{% if x_labels|length + 1 - loop.index > 0 %}{% endif %}{% endfor %}@{}}
\toprule
& {% for x_label in x_labels %} {{x_label}} {% if x_labels|length - loop.index > 0 %} & {% else %} \\  {% endif %} {% endfor %} 
\midrule
{% for y_label in y_labels %} {{y_label}} & {% for x_label in x_labels %} {{ table_data[x_label][y_label] }} {% if y_labels|length - loop.index > 0 %} & {% endif %} {% endfor %} {% if x_labels|length - loop.index > 0 %} \\ {% endif %}  
{% endfor %}
\bottomrule
\end{tabularx}
\caption{\label{tab:}}
\end{table}""")
    
#tmpl_rst_h = """
#{% macro dashfill() -%}
#{% for item in range(x_labels|sort(attribute="count")|first|count + 4 ) %}-{% endfor %}
#{%- endmacro %}
#{% macro spacefill(value) -%}
#{{value}}{% for item in range(x_labels|sort(attribute="count")|first|count + 4 - value|string|length ) %} {% endfor %}
#{%- endmacro %}
#{% macro tableline() -%}
#+{% for y_label in range(y_labels|length + 1) %}{{dashfill()}}+{% endfor %}
#{%- endmacro %}
#{{tableline()}}
#|{{spacefill()}}|{% for y_label in y_labels %}{{spacefill(y_label)}}|{% endfor %}
#{{tableline()}}
#{% for x_label in x_labels %}|{{spacefill(x_label)}}{% for y_label in y_labels %}|{{spacefill(table_data[x_label][y_label])}}{% endfor %}|
#{{tableline()}}
#{% endfor %}
#"""
    
tmpl_rst_h = lambda maxlength : Template("""
{% macro dashfill() -%}
{% for item in range(""" + str(maxlength) + """ + 4 ) %}-{% endfor %}
{%- endmacro %}
{% macro spacefill(value) -%}
{{value}}{% for item in range(""" + str(maxlength) + """ + 4 - value|string|length ) %} {% endfor %}
{%- endmacro %}
{% macro tableline() -%}
+{% for y_label in range(y_labels|length + 1) %}{{dashfill()}}+{% endfor %}
{%- endmacro %}
{{tableline()}}
|{{spacefill()}}|{% for y_label in y_labels %}{{spacefill(y_label)}}|{% endfor %}
{{tableline()}}
{% for x_label in x_labels %}|{{spacefill(x_label)}}{% for y_label in y_labels %}|{{spacefill(table_data[x_label][y_label])}}{% endfor %}|
{{tableline()}}
{% endfor %}
""") 
    
tmpl_rst_v = lambda maxlength : Template("""
{% macro dashfill() -%}
{% for item in range(""" + str(maxlength) + """ + 4 ) %}-{% endfor %}
{%- endmacro %}
{% macro spacefill(value) -%}
{{value}}{% for item in range(""" + str(maxlength) + """ + 4 - value|string|length ) %} {% endfor %}
{%- endmacro %}
{% macro tableline() -%}
+{% for x_label in range(x_labels|length + 1) %}{{dashfill()}}+{% endfor %}
{%- endmacro %}
{{tableline()}}
|{{spacefill()}}|{% for x_label in x_labels %}{{spacefill(x_label)}}|{% endfor %}
{{tableline()}}
{% for y_label in y_labels %}|{{spacefill(y_label)}}{% for x_label in x_labels %}|{{spacefill(table_data[x_label][y_label])}}{% endfor %}|
{{tableline()}}
{% endfor %}
""")



class TinyTable(object):
    """
        This class allows to piecewise assemble a table and render it then into wiki or html syntax.

        Internally each cell is indexed by a x_label and a y_label. For the piecewise construction it
        offers the 'add' method. This adds all cells for a given x_label. The order of the x_labels
        can be determined by order of the add calls. The y_labels are alphabetically sorted.

        The render method will convert the table into wiki or html syntax. There one can decide
        if the y_labels should be horizontally or vertically layouted
    """

    def __init__(self):
        self.x_labels = []
        self.label_data = dict()

    @property
    def y_labels(self):
        result = set()
        for i in self.label_data.keys():
            for j in self.label_data[i].keys():
                result.add(j)
        return list(result)

    def add(self, label, **kwargs):
        """
            Add a row/column. label will be added to the x_labels, 
            the parameters in **kwargs will be added to y_labels.
        """
        self.x_labels.append(label)
        self.label_data[label] = kwargs

    def render(self, layout="v", format="wiki",format_cell=lambda x: x):
        """
            render this table into a given output format

            layout may be either "h" or "v", for horizontal or vertical layout
            format may bei either "wiki","html" or "rst"
            format_cell is a function which can be used to format the the cell content, e.g. lambda x : "%1.2e" %str(x)
        """
        table_data = defaultdict(dict)
        y_labels = sorted(self.y_labels)
   
        for x_label in self.x_labels:
            for y_label in y_labels:
                table_data[x_label][y_label] = format_cell(self.label_data[x_label].get(y_label, ""))
        

        max_cell_length = max( max(map(len,self.x_labels)),max(map(len,self.y_labels)))
        max_cell_length = max(max_cell_length,max(map(len,[map(len,x.values()) for x in table_data.values()])))
        if format == "html":
            if layout == "v":
                return tmpl_html_v.render(table_data=table_data, 
                                         x_labels = self.x_labels,
                                         y_labels = y_labels)
            elif layout == "h":
                return tmpl_html_h.render(table_data=table_data, 
                                         x_labels = self.x_labels,
                                         y_labels = y_labels)
            else:
                raise ValueError("layout not supported")
        elif format == "wiki":
            if layout == "v":
                return tmpl_wiki_v.render(table_data=table_data, 
                                         x_labels = self.x_labels,
                                         y_labels = y_labels)
            elif layout == "h":
                return tmpl_wiki_h.render(table_data=table_data, 
                                         x_labels = self.x_labels,
                                         y_labels = y_labels)
            else:
                raise ValueError("layout not supported")
        elif format == "latex":
            if layout == "v":
                return tmpl_latex_v.render(table_data=table_data, 
                                         x_labels = self.x_labels,
                                         y_labels = y_labels)
            elif layout == "h":
                return tmpl_latex_h.render(table_data=table_data, 
                                         x_labels = self.x_labels,
                                         y_labels = y_labels)
            else:
                raise ValueError("layout not supported")
        elif format == "latex_pretty":
            if layout == "v":
                return tmpl_latex_pretty_v.render(table_data=table_data, 
                                         x_labels = self.x_labels,
                                         y_labels = y_labels)
            elif layout == "h":
                return tmpl_latex_pretty_h.render(table_data=table_data, 
                                         x_labels = self.x_labels,
                                         y_labels = y_labels)
            else:
                raise ValueError("layout not supported")
        elif format == "rst":
            if layout == "v":
                return tmpl_rst_v(max_cell_length).render(table_data=table_data, 
                                         x_labels = self.x_labels,
                                         y_labels = y_labels)
            elif layout == "h":
                return tmpl_rst_h(max_cell_length).render(table_data=table_data, 
                                         x_labels = self.x_labels,
                                         y_labels = y_labels)
            else:
                raise ValueError("layout not supported")
        else:
            raise ValueError("format not supported")

        

if __name__ == "__main__":
    t = TinyTable()
    t.add("no cut", data=1, nue=1, numu=1)
    t.add("cut 1" , data=3.23463e-5, nue=1.92346345e-11, numu=2.12351243234e-12)
    t.add("cut 2" , data=.1, nue=.5, numu=.5)

    print 80*"-"
    print "vertical alignment"
    print 80*"-"
    print t.render(layout="v", format="wiki",format_cell = lambda x: "%1.2e" %x)
    print 80*"-"
    print "horizontal alignment"
    print 80*"-"
    print t.render(layout="h", format="wiki",format_cell = lambda x: "%1.2e" %x)
    print 80*"-"
    print "latex"
    print t.render(layout="h", format="latex",format_cell = lambda x: "%1.2e" %x)
    print t.render(layout="v", format="latex",format_cell = lambda x:     "%1.2e" %x)
    print 80*"-"
    print "latex-pretty"
    print t.render(layout="h", format="latex_pretty",format_cell = lambda x: "%1.2e" %x)
    print t.render(layout="v", format="latex_pretty",format_cell = lambda x:     "%1.2e" %x)
    print 80*"-"
    print "rst"
    print t.render(layout="h", format="rst",format_cell = lambda x:     "%1.2e" %x)
    print t.render(layout="v", format="rst",format_cell = lambda x:     "%1.2e" %x)

