debugger;
table = document.getElementById("searchSharesListTable-SSE365").parentElement;

var data = { telecom : [],
             energy: [],
             materials: [],
             industrials: [],
             consumer_goods: [],
             consumer_services: [],
             healthcare: [],
             utilities: [],
             financials: [],
             information_technology: [],
             realestate: [] }

for(var i=0;i<table.rows.length;i++){
  row = table.rows[i];
  className = row.children[0].className;
  id = row.id.replace('searchSharesListTable-', '');
  name = row.title.split(" - ")[1].replaceAll(" ", "_");
  data[className].push({id: id, name: name});
}
