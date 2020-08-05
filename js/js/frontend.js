function Point(x, y) {
    this.x = x;
    this.y = y;
}

var PolyLine = function(canvas) {
    let self = this;

    self.canvas = canvas;

    self.x = 0;
    self.y = 0;

    self.poly = null;
    self.polyPoints = [];
    self.lines = [];
    self.lineCounter = 0;
    self.drawingObject = {};
    self.drawingObject.type = "empty";
    self.drawingObject.background = "";
    self.drawingObject.border = "";
    
    fabric.util.addListener(window, 'dblclick', function(){
        if (self.drawingObject.type == "empty") { 
            self.drawingObject.type = "created";
            self.lines.forEach(function(value, index, ar){
                self.canvas.remove(value);
            });
            //canvas.remove(lines[lineCounter - 1]);
            self.poly = self.makepoly();
            self.canvas.add(self.poly);
            self.canvas.renderAll();
        } else {
            self.clear();
        }
      
        console.log("double click");
    });

    self.canvas.on('mouse:down', function (options) {
        if (self.drawingObject.type == "empty") {
            self.canvas.selection = false;
            self.setStartingPoint(options); // set x,y
            self.polyPoints.push(new Point(self.x, self.y));
            var points = [self.x, self.y, self.x, self.y];
            self.lines.push(new fabric.Line(points, {
                strokeWidth: 3,
                selectable: false,
                stroke: 'red'
            }));
            self.canvas.add(self.lines[self.lineCounter]);
            self.lineCounter++;
            self.canvas.on('mouse:up', function (options) {
                self.canvas.selection = true;
            });
        }
    });

    self.canvas.on('mouse:move', function (options) {
        if (self.lines[0] !== null && self.lines[0] !== undefined && self.drawingObject.type == "empty") {
            self.setStartingPoint(options);
            self.lines[self.lineCounter - 1].set({
                x2: self.x,
                y2: self.y
            });
            self.canvas.renderAll();
        }
    });
    
    $("#poly").click(function () {
        self.clear();
    });
}

PolyLine.prototype = {
    clear: function() {
        let self = this;
        self.canvas.clear();
        self.poly = null;
        self.polyPoints.length = 0;
        self.lines.length = 0;
        self.lineCounter = 0;
        self.drawingObject = {};
        self.drawingObject.type = "empty";
        self.canvas.renderAll();
    },

    findTopPaddingForpoly: function() {
        let self = this;
        var result = 999999;
        for (var f = 0; f < self.lineCounter; f++) {
            if (self.polyPoints[f].y < result) {
                result = self.polyPoints[f].y;
            }
        }
        return Math.abs(result);
    },

    findLeftPaddingForpoly: function() {
        let self = this;
        var result = 999999;
        for (var i = 0; i < self.lineCounter; i++) {
            if (self.polyPoints[i].x < result) {
                result = self.polyPoints[i].x;
            }
        }
        return Math.abs(result);
    },

    makepoly: function() {
        let self = this;
        var left = self.findLeftPaddingForpoly();
        var top  = self.findTopPaddingForpoly();
        self.polyPoints[self.polyPoints.length - 1] = new Point(self.polyPoints[0].x, self.polyPoints[0].y);
        // self.polyPoints.push(new Point(self.polyPoints[0].x, self.polyPoints[0].y))
        var poly = new fabric.Polyline(self.polyPoints, {
            fill: 'rgba(0,0,0,0)',
            stroke:'#58c'
        });

        poly.set({    
            left: left,
            top: top,
        });

        poly.on("modified", function () {
            var matrix = this.calcTransformMatrix();
            self.polyPoints = this.get("points")
            .map(function(p){
                return new fabric.Point(
                p.x - poly.pathOffset.x,
                p.y - poly.pathOffset.y);
            })
            .map(function(p){
            return fabric.util.transformPoint(p, matrix);
            });

            // display circles for debugging
            // var circles = []
            // for (var i = 0; i < self.polyPoints.length - 1; i++) {
            //     var p = self.polyPoints[i];
            //     var circle = new fabric.Circle({
            //         left: p.x,
            //         top: p.y,
            //         radius: 3,
            //         fill: "red",
            //         originX: "center",
            //         originY: "center",
            //         hasControls: false,
            //         hasBorders: false,
            //         selectable: false
            //       });
            //     // self.canvas.add(circle);
            //     circles.push(circle);
            // }
            // self.canvas.clear().add(self.poly).add.apply(self.canvas, circles).setActiveObject(self.poly).renderAll();

        });

        return poly;
    },

    setStartingPoint: function(options) {
        let self = this;
        var offset = $('#canvas-tools').offset();
        self.x = options.e.pageX - offset.left;
        self.y = options.e.pageY - offset.top;
    }
}

// canvas Drawing
var canvas = new fabric.Canvas('canvas-tools');
var polyLine = new PolyLine(canvas);
