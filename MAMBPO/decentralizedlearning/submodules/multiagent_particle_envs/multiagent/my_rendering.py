"""
2D rendering framework adapted for pyglet 2.0.x
移行させたいがむずい
"""
from __future__ import division
import os
import six
import sys
import math
import numpy as np

from gymnasium import error
import pyglet
from pyglet import gl

if "Apple" in sys.version:
    if 'DYLD_FALLBACK_LIBRARY_PATH' in os.environ:
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] += ':/usr/lib'
        # Avoid bug on some systems

RAD2DEG = 57.29577951308232

def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            'Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec)
        )

class Viewer(object):
    def __init__(self, width, height, display=None):
        display = get_display(display)

        self.width = width
        self.height = height

        self.window = pyglet.window.Window(width=width, height=height, display=display)
        
        # Register close event
        @self.window.event
        def on_close():
            self.window_closed_by_user()

        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()

        gl.glEnable(gl.GL_BLEND)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        gl.glLineWidth(2.0)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        if self.window:
            self.window.close()

    def window_closed_by_user(self):
        self.close()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width/(right-left)
        scaley = self.height/(top-bottom)
        self.transform = Transform(
            translation=(-left*scalex, -bottom*scaley),
            scale=(scalex, scaley)
        )

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array=False):
        gl.glClearColor(1,1,1,1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()
        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            # Use frombuffer instead of fromstring
            raw_data = image_data.get_data('RGBA', buffer.width * 4)
            arr = np.frombuffer(raw_data, dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1,:,0:3]
        self.window.flip()
        self.onetime_geoms = []
        return arr

    def draw_circle(self, radius=10, res=30, filled=True, **attrs):
        geom = make_circle(radius=radius, res=res, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        geom = make_polygon(v=v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v, **attrs):
        geom = make_polyline(v=v)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def get_array(self):
        self.window.flip()
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        self.window.flip()
        raw_data = image_data.get_data('RGBA', self.width * 4)
        arr = np.frombuffer(raw_data, dtype=np.uint8)
        arr = arr.reshape(self.height, self.width, 4)
        return arr[::-1,:,0:3]

def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])

class Geom(object):
    def __init__(self):
        self._color=Color((0, 0, 0, 1.0))
        self.attrs = [self._color]
    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()
    def render1(self):
        raise NotImplementedError
    def add_attr(self, attr):
        self.attrs.append(attr)
    def set_color(self, r, g, b, alpha=1):
        self._color.vec4 = (r, g, b, alpha)

class Attr(object):
    def enable(self):
        raise NotImplementedError
    def disable(self):
        pass

class Transform(Attr):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1,1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)
    def enable(self):
        gl.glPushMatrix()
        gl.glTranslatef(self.translation[0], self.translation[1], 0)
        gl.glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        gl.glScalef(self.scale[0], self.scale[1], 1)
    def disable(self):
        gl.glPopMatrix()
    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))
    def set_rotation(self, new):
        self.rotation = float(new)
    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))

class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4
    def enable(self):
        gl.glColor4f(*self.vec4)

class LineStyle(Attr):
    def __init__(self, style):
        self.style = style
    def enable(self):
        gl.glEnable(gl.GL_LINE_STIPPLE)
        gl.glLineStipple(1, self.style)
    def disable(self):
        gl.glDisable(gl.GL_LINE_STIPPLE)

class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke
    def enable(self):
        gl.glLineWidth(self.stroke)

class Point(Geom):
    def __init__(self):
        Geom.__init__(self)
    def render1(self):
        gl.glBegin(gl.GL_POINTS)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glEnd()

class FilledPolygon(Geom):
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v
    def render1(self):
        if   len(self.v) == 4 : gl.glBegin(gl.GL_QUADS)
        elif len(self.v)  > 4 : gl.glBegin(gl.GL_POLYGON)
        else: gl.glBegin(gl.GL_TRIANGLES)
        for p in self.v:
            gl.glVertex3f(p[0], p[1],0)
        gl.glEnd()

        # Outline
        color = (self._color.vec4[0] * 0.5, self._color.vec4[1] * 0.5, self._color.vec4[2] * 0.5, self._color.vec4[3] * 0.5)
        gl.glColor4f(*color)
        gl.glBegin(gl.GL_LINE_LOOP)
        for p in self.v:
            gl.glVertex3f(p[0], p[1],0)
        gl.glEnd()

def make_circle(radius=10, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2*math.pi*i / res
        points.append((math.cos(ang)*radius, math.sin(ang)*radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)

def make_polygon(v, filled=True):
    if filled: return FilledPolygon(v)
    else: return PolyLine(v, True)

def make_polyline(v):
    return PolyLine(v, False)

def make_capsule(length, width):
    l, r, t, b = 0, length, width/2, -width/2
    box = make_polygon([(l,b), (l,t), (r,t), (r,b)])
    circ0 = make_circle(width/2)
    circ1 = make_circle(width/2)
    circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom

class Compound(Geom):
    def __init__(self, gs):
        Geom.__init__(self)
        self.gs = gs
        for g in self.gs:
            g.attrs = [a for a in g.attrs if not isinstance(a, Color)]
    def render1(self):
        for g in self.gs:
            g.render()

class PolyLine(Geom):
    def __init__(self, v, close):
        Geom.__init__(self)
        self.v = v
        self.close = close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)
    def render1(self):
        gl.glBegin(gl.GL_LINE_LOOP if self.close else gl.GL_LINE_STRIP)
        for p in self.v:
            gl.glVertex3f(p[0], p[1],0)
        gl.glEnd()
    def set_linewidth(self, x):
        self.linewidth.stroke = x

class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        gl.glBegin(gl.GL_LINES)
        gl.glVertex2f(*self.start)
        gl.glVertex2f(*self.end)
        gl.glEnd()

class Image(Geom):
    def __init__(self, fname, width, height):
        Geom.__init__(self)
        self.width = width
        self.height = height
        img = pyglet.image.load(fname)
        self.img = img
        self.flip = False
    def render1(self):
        self.img.blit(-self.width/2, -self.height/2, width=self.width, height=self.height)

# ================================================================

class SimpleImageViewer(object):
    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display
    def imshow(self, arr):
        if self.window is None:
            height, width, channels = arr.shape
            self.window = pyglet.window.Window(width=width, height=height, display=self.display)
            self.width = width
            self.height = height
            self.isopen = True

            @self.window.event
            def on_close():
                self.close()

        assert arr.shape == (self.height, self.width, 3), "Wrong image shape"
        image = pyglet.image.ImageData(self.width, self.height, 'RGB', arr.tobytes(), pitch=self.width * -3)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0,0)
        self.window.flip()
    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False
    def __del__(self):
        self.close()
