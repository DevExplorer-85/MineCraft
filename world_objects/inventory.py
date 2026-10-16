# name: (label, color, placeable block id)
ITEMS = {
    'log':          ('Log',            (115, 82, 46),   0),
    'planks':       ('Planks',         (196, 156, 92),  11),
    'stick':        ('Stick',          (150, 112, 60),  0),
    'dirt':         ('Dirt',           (115, 78, 46),   3),
    'sand':         ('Sand',           (230, 210, 140), 1),
    'stone':        ('Stone',          (128, 128, 128), 4),
    'raw_pork':     ('Raw Pork',       (240, 150, 150), 0),
    'raw_beef':     ('Raw Beef',       (190, 60, 60),   0),
    'raw_mutton':   ('Raw Mutton',     (215, 105, 105), 0),
    'raw_venison':  ('Raw Venison',    (170, 70, 70),   0),
    'cooked_pork':  ('Cooked Pork',    (200, 130, 80),  0),
    'cooked_beef':  ('Steak',          (130, 70, 40),   0),
    'cooked_mutton': ('Cooked Mutton', (165, 95, 60),   0),
    'cooked_venison': ('Cooked Venison', (140, 80, 50), 0),
    'wood_axe':     ('Wooden Axe',     (200, 160, 90),  0),
    'wood_sword':   ('Wooden Sword',   (210, 175, 110), 0),
    'wood_pickaxe': ('Wooden Pickaxe', (190, 150, 85),  0),
}

# (label, inputs, (output, count))
RECIPES = [
    ('Planks x4',      {'log': 1},                      ('planks', 4)),
    ('Sticks x4',      {'planks': 2},                   ('stick', 4)),
    ('Wooden Axe',     {'planks': 3, 'stick': 2},       ('wood_axe', 1)),
    ('Wooden Sword',   {'planks': 2, 'stick': 1},       ('wood_sword', 1)),
    ('Wooden Pickaxe', {'planks': 3, 'stick': 2},       ('wood_pickaxe', 1)),
    ('Cook Pork',      {'raw_pork': 1, 'planks': 1},    ('cooked_pork', 1)),
    ('Cook Beef',      {'raw_beef': 1, 'planks': 1},    ('cooked_beef', 1)),
    ('Cook Mutton',    {'raw_mutton': 1, 'planks': 1},  ('cooked_mutton', 1)),
    ('Cook Venison',   {'raw_venison': 1, 'planks': 1}, ('cooked_venison', 1)),
]


class Inventory:
    def __init__(self):
        self.c = {}
        self.order = []
        self.version = 0

    def count(self, name):
        return self.c.get(name, 0)

    def add(self, name, k=1):
        if self.count(name) == 0:
            self.order.append(name)
        self.c[name] = self.count(name) + k
        self.version += 1

    def remove(self, name, k=1):
        self.c[name] -= k
        if self.c[name] <= 0:
            self.c[name] = 0
            self.order.remove(name)
        self.version += 1

    def has(self, need):
        return all(self.count(k) >= v for k, v in need.items())

    def craft(self, recipe):
        _, need, (out, n) = recipe
        if not self.has(need):
            return False
        for k, v in need.items():
            self.remove(k, v)
        self.add(out, n)
        return True

    def items(self):
        return [(k, self.c[k]) for k in self.order]