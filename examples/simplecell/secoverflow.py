"""Test section stack overflow"""

morphology_path = 'simple.swc'

import neuron

neuron.h.load_file('stdrun.hoc')
neuron.h.load_file('import3d.hoc')


def get_icell():
    """Get instantiated cell"""

    if hasattr(neuron.h, 'cell'):
        return neuron.h.cell()
    else:
        template_content = 'begintemplate %s\n' \
            'objref all, apical, basal, somatic, axonal\n' \
            'proc init() {\n' \
            'all 	= new SectionList()\n' \
            'somatic = new SectionList()\n' \
            'basal 	= new SectionList()\n' \
            'apical 	= new SectionList()\n' \
            'axonal 	= new SectionList()\n' \
            'forall delete_section()\n' \
            '}\n' \
            'create soma[1], dend[1], apic[1], axon[1]\n' \
            'endtemplate %s\n' % ('cell', 'cell')

        neuron.h(template_content)

        template_function = getattr(neuron.h, 'cell')

        return template_function()


for index in range(100):
    print index

    icell = get_icell()
    print icell

    extension = morphology_path.split('.')[-1]

    if extension.lower() == 'swc':
        imorphology = neuron.h.Import3d_SWC_read()
    elif extension.lower() == 'asc':
        imorphology = neuron.h.Import3d_Neurolucida3()
    else:
        raise Exception("Unknown filetype: %s" % extension)

    # TODO this is to get rid of stdout print of neuron
    # probably should be more intelligent here, and filter out the
    # lines we don't want
    neuron.h.hoc_stdout('/dev/null')
    imorphology.input(morphology_path)
    neuron.h.hoc_stdout()

    morphology_importer = neuron.h.Import3d_GUI(imorphology, 0)

    morphology_importer.instantiate(icell)
    neuron.h.secname()
    del icell
