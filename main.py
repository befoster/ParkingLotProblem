from shapely.geometry import Polygon, MultiPolygon, LineString, Point, GeometryCollection
import pygame as pg


CAP_STYLE = 1
JOIN_STYLE = 1


class Draw:
    def __init__(self, polygon, iter_per_sec=240, fps=60, dim=(720, 720)):
        # Initialize pygame settings
        pg.init()
        self.screen = pg.display.set_mode(dim)
        self.dim = dim
        self.fps = fps
        self.clock = pg.time.Clock()
        self.speed = iter_per_sec / fps
        self.counter = 0

        # Scale polygon to fit screen nicely
        min_x, min_y, max_x, max_y = polygon.buffer(24, cap_style=CAP_STYLE, join_style=JOIN_STYLE).bounds
        self.scale = min(self.dim[0] * 0.9 / (max_x - min_x), self.dim[1] * 0.9 / (max_y - min_y))
        center_x, center_y = self.scale * (min_x + max_x) / 2, self.scale * (min_y + max_y) / 2
        self.offset = self.dim[0] / 2 - center_x, self.dim[1] / 2 - center_y

        # Base polygon is the same for all iterations, so only draw it once
        self.polygon_surf = pg.Surface(dim)
        self.polygon_surf.fill((255, 255, 255))
        self.draw_poly(self.polygon_surf, polygon, (0, 0, 0), 4)

    # Get list of (x, y) coords, scale and translate them, and draw them
    def draw_poly(self, surf, polygon, color, fill):
        x, y = polygon.exterior.xy
        pts = [(x[i] * self.scale + self.offset[0], y[i] * self.scale + self.offset[1]) for i in range(len(x)-1)]
        pg.draw.polygon(surf, color, pts, fill)

    # Skip frames or draw for multiple frames, depending on iteration speed and fps
    def pre_draw(self):
        self.counter += self.speed
        if self.counter < 1:
            return False
        self.counter -= 1
        while self.counter >= 1:
            if pg.event.peek(pg.QUIT):
                pg.quit()
                return False
            pg.event.clear()
            self.clock.tick(self.fps)
            self.counter -= 1
        return True

    # Draws list of rectangles to the screen
    def draw(self, rects, override_skip=False):
        if (not override_skip) and self.pre_draw():
            self.screen.blit(self.polygon_surf, (0, 0))
            for rect in rects:
                self.draw_poly(self.screen, rect.buffer(24), (200, 0, 0), 2)
            for rect in rects:
                self.draw_poly(self.screen, rect, (0, 0, 200), 0)
            if pg.event.peek(pg.QUIT):
                pg.quit()
                return
            pg.event.clear()
            self.clock.tick(self.fps)
            pg.display.update()


def midpoint_1(x1, y1, x2, y2):
    return (x1+x2)/2, (y1+y2)/2


def midpoint_2(p1, p2):
    return (p1[0]+p2[0])/2, (p1[1]+p2[1])/2


def extend_line(p, x1, y1, x2, y2):
    min_x, min_y, max_x, max_y = p.bounds
    while not ((min(x1, x2) < min_x and max(x1, x2) > max_x) or (min(y1, y2) < min_y and max(y1, y2) > max_y)):
        dx, dy = x2 - x1, y2 - y1
        x1 -= dx
        x2 += dx
        y1 -= dy
        y2 += dy
    return x1, y1, x2, y2


def find_bounds(p, x1, y1, x2, y2, m):
    line = LineString([(x1, y1), (x2, y2)])
    divs = {m, (x1, y1), (x2, y2)}
    divs.update(get_xy_coords(p.intersection(line)))
    pts = sorted(sorted(divs, key=lambda pt: pt[0]), key=lambda pt: pt[1])
    j = pts.index(m) + 1
    while j + 1 < len(pts):
        pt = Point((pts[j][0] + pts[j + 1][0]) / 2, (pts[j][1] + pts[j + 1][1]) / 2)
        if p.touches(pt) or p.contains(pt):
            j += 1
        else:
            break
    pt1 = pts[j]
    j = pts.index(m) - 1
    while j - 1 >= 0:
        pt = Point((pts[j][0] + pts[j - 1][0]) / 2, (pts[j][1] + pts[j - 1][1]) / 2)
        if p.touches(pt) or p.contains(pt):
            j -= 1
        else:
            break
    pt2 = pts[j]
    return pt1, pt2


def completely_contained(line, p):
    return line.equals(p.intersection(line))


def make_rect(d1, d2, side):
    x1, y1 = side.parallel_offset(d1, 'right').coords.xy
    x2, y2 = side.parallel_offset(d2, 'right').coords.xy
    return Polygon(order_pts((x1[0], y1[0]), (x1[1], y1[1]), (x2[0], y2[0]), (x2[1], y2[1])))


def get_xy_coords(p):
    p_type = p.geometryType()
    if 'Multi' in p_type or 'Collection' in p_type:
        pts = list()
        for g in p:
            pts += get_xy_coords(g)[:]
        return pts
    elif p_type == 'Polygon':
        return get_xy_coords(p.exterior)
    return p.coords[:]


def order_pts(pt1, pt2, pt3, pt4):
    if Point(pt2).distance(Point(pt3)) < Point(pt2).distance(Point(pt4)):
        return [pt1, pt2, pt3, pt4]
    else:
        return [pt1, pt2, pt4, pt3]


def find_rects(p, pt1, pt2, d, w, h, side):
    if w > d:
        return list()
    pl = LineString([pt1, pt2]).parallel_offset(h, side)
    if pl.disjoint(p):
        return list()
    pl_coords = pl.coords.xy
    rect_pts = order_pts(pt1, pt2, (pl_coords[0][1], pl_coords[1][1]), (pl_coords[0][0], pl_coords[1][0]))
    big_rect = Polygon(rect_pts)
    side = LineString([rect_pts[0], rect_pts[3]])
    d_vals = {side.distance(Point(coord)) for coord in get_xy_coords(big_rect.intersection(p)) if side.distance(Point(coord)) > 0.0000001}
    d_vals.add(0)
    d_vals.add(w)
    d_vals = sorted(d_vals)
    segmented = list()
    d1, d2 = None, None
    for i in range(1, len(d_vals)):
        if completely_contained(side.parallel_offset((d_vals[i]+d_vals[i-1])/2, 'right'), p):
            d2 = d_vals[i]
            if d1 is None:
                d1 = d_vals[i-1]
        else:
            if d1 is not None:
                segmented.append((d1, d2))
                d1, d2 = None, None
    if d1 is not None:
        segmented.append((d1, d2))
    res = list()
    for d1, d2 in segmented:
        if d2 - d1 == w:
            res.append(make_rect(d1, d2, side))
        elif d2 - d1 > w:
            res.append(make_rect(d1, d1+w, side))
            res.append(make_rect(d2-w, d2, side))
    return res


def h_rects(p, pt1, pt2, d):
    k = 1
    s = 8.5
    prev_len = -1
    res = list()
    while s <= d and prev_len != len(res):
        prev_len = len(res)
        res += [(rect, k) for rect in find_rects(p, pt1, pt2, d, s, 18, 'left')]
        res += [(rect, k) for rect in find_rects(p, pt1, pt2, d, s, 18, 'right')]
        res += [(rect, 2 * k) for rect in find_rects(p, pt1, pt2, d, s, 36, 'left')]
        res += [(rect, 2 * k) for rect in find_rects(p, pt1, pt2, d, s, 36, 'right')]
        k += 1
        s += 8.5
    return res


def v_rects(p, pt1, pt2, d, side, w):
    prev_len = -1
    k = 1
    res = list()
    while prev_len != len(res):
        prev_len = len(res)
        res += [(rect, w*k) for rect in find_rects(p, pt1, pt2, d, 18*w, 8.5*k, side)]
        k += 1
    return res


def one_angle(p, x1, y1, x2, y2):
    x3, y3, x4, y4 = extend_line(p, x1, y1, x2, y2)
    pt1, pt2 = find_bounds(p, x3, y3, x4, y4, midpoint_1(x1, y1, x2, y2))
    d = Point(pt1).distance(Point(pt2))
    res = list()
    if d >= 8.5:
        res += h_rects(p, pt1, pt2, d)
        if d >= 18:
            res += v_rects(p, pt1, pt2, d, 'left', 1)
            res += v_rects(p, pt1, pt2, d, 'right', 1)
            if d >= 36:
                res += v_rects(p, pt1, pt2, d, 'left', 2)
                res += v_rects(p, pt1, pt2, d, 'right', 2)
    return res


# Get a list of rectangles (and their weights) that can be drawn in the current polygon
def get_possible_rects(p):
    if p.area < 153:
        return list()
    x, y = p.exterior.xy
    res = list()
    for i in range(1, len(x)):
        res += one_angle(p, x[i-1], y[i-1], x[i], y[i])
    return res


class Node:
    def __init__(self, mp, rect=None, weight=None, current_rects=None, current_weight=None):
        if rect is None:
            # Root node
            self.mp = mp
            self.rect_list = list()
            self.weight = 0
        else:
            # Remove rect from current polygon and add its weight
            self.mp = mp.difference(rect.buffer(24, cap_style=CAP_STYLE, join_style=JOIN_STYLE))
            self.rect_list = current_rects + [rect]
            self.weight = current_weight + weight
        # Find max possible weight of future rects
        if self.mp.geometryType() == 'Polygon':
            self.mp = MultiPolygon([self.mp])
        self.potential = self.weight
        for p in self.mp:
            self.potential += p.area // 153

    # Get a list of potential child nodes
    def get_next_rects(self, draw=None):
        if draw is not None:
            draw.draw(self.rect_list)
        new_rects = list()
        to_remove = list()
        if self.mp.geometryType() == 'Polygon':
            self.mp = MultiPolygon([self.mp])
        for p in self.mp:
            more_rects = get_possible_rects(p)
            if not more_rects:
                to_remove.append(p)
            new_rects += more_rects[:]
        for p in to_remove:
            self.mp = self.mp.difference(p)
        nodes = [Node(self.mp, rect, weight, self.rect_list, self.weight) for rect, weight in new_rects]
        nodes.sort(reverse=True, key=lambda n: n.potential)
        return nodes


# Run the DFS
def main(polygon=None):
    if not polygon:
        # Get the input polygon
        input_vals = input('Point list (x1,y1,x2,y2,x3,y3,...):\n')
        vals = [float(k.strip()) for k in input_vals.split(',')]
        polygon = Polygon([(vals[i-1], vals[i]) for i in range(1, len(vals), 2)])
    # Start the DFS
    draw = Draw(polygon)
    best_countdown = 0
    queue = [Node(MultiPolygon([polygon]))]
    best = queue[0]
    while queue:
        node = queue.pop(0)
        if node.weight > best.weight:
            best = node
            print('New best found: Weight =', best.weight)
            best_countdown = draw.speed * draw.fps * 1
        queue += node.get_next_rects()
        if best_countdown > 0:
            best_countdown -= 1
            draw.draw(best.rect_list)
        else:
            draw.draw(node.rect_list)
        queue.sort(reverse=True, key=lambda n: n.potential)
        queue.sort(reverse=True, key=lambda n: n.weight)
        # Remove nodes which have a lower potential weight than the current best
        while queue and queue[-1].potential <= best.weight:
            queue.pop()
    draw.draw(best.rect_list)
    input('Search complete')
    pg.quit()


if __name__ == '__main__':
    main()
