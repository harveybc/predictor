<div class="view">

	<b><?php echo CHtml::encode($data->getAttributeLabel('titulo')); ?>:</b>
        <?php echo CHtml::link(CHtml::encode($data->titulo), array('view', 'id'=>$data->id)); ?>
	
	<br />
        
	<b><?php echo CHtml::encode($data->getAttributeLabel('ubicacionT')); ?>:</b>
	<?php echo CHtml::encode($data->ubicacionT); ?>
	<br />

	
	<b><?php echo CHtml::encode($data->getAttributeLabel('numPedido')); ?>:</b>
	<?php echo CHtml::encode($data->numPedido); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('numComision')); ?>:</b>
	<?php echo CHtml::encode($data->numComision); ?>
	<br />

	

	<?php /*
         * 
         * <b><?php echo CHtml::encode($data->getAttributeLabel('cerveceria')); ?>:</b>
	<?php echo CHtml::encode($data->cerveceria); ?>
	<br />
         * 
         * 
	<b><?php echo CHtml::encode($data->getAttributeLabel('descripcion')); ?>:</b>
	<?php echo CHtml::encode($data->descripcion); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('tipoContenido')); ?>:</b>
	<?php echo CHtml::encode($data->tipoContenido); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('version')); ?>:</b>
	<?php echo CHtml::encode($data->version); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('medio')); ?>:</b>
	<?php echo CHtml::encode($data->medio); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('idioma')); ?>:</b>
	<?php echo CHtml::encode($data->idioma); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('disponibles')); ?>:</b>
	<?php echo CHtml::encode($data->disponibles); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('existencias')); ?>:</b>
	<?php echo CHtml::encode($data->existencias); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('modulo')); ?>:</b>
	<?php echo CHtml::encode($data->modulo); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('columna')); ?>:</b>
	<?php echo CHtml::encode($data->columna); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('fila')); ?>:</b>
	<?php echo CHtml::encode($data->fila); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('documento')); ?>:</b>
	<?php echo CHtml::encode($data->documento); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('ruta')); ?>:</b>
	<?php echo CHtml::encode($data->ruta); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('fechaCreacion')); ?>:</b>
	<?php echo CHtml::encode($data->fechaCreacion); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('fechaRecepcion')); ?>:</b>
	<?php echo CHtml::encode($data->fechaRecepcion); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('autores')); ?>:</b>
	<?php echo CHtml::encode($data->autores); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('usuario')); ?>:</b>
	<?php echo CHtml::encode($data->usuario); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('revisado')); ?>:</b>
	<?php echo CHtml::encode($data->revisado); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('userRevisado')); ?>:</b>
	<?php echo CHtml::encode($data->userRevisado); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('fechaRevisado')); ?>:</b>
	<?php echo CHtml::encode($data->fechaRevisado); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('eliminado')); ?>:</b>
	<?php echo CHtml::encode($data->eliminado); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('secuencia')); ?>:</b>
	<?php echo CHtml::encode($data->secuencia); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('ordenSecuencia')); ?>:</b>
	<?php echo CHtml::encode($data->ordenSecuencia); ?>
	<br />
         * <b><?php echo CHtml::encode($data->getAttributeLabel('ISBN')); ?>:</b>
	<?php echo CHtml::encode($data->ISBN); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('EAN13')); ?>:</b>
	<?php echo CHtml::encode($data->EAN13); ?>
	<br />

	*/ ?>

</div>
