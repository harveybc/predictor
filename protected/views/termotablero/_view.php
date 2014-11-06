<div class="view">
    
         <b><?php echo CHtml::encode($data->getAttributeLabel('TAG')); ?>:</b>
	<?php echo CHtml::link(CHtml::encode($data->TAG), array('view', 'id'=>$data->id)); ?>
	<br />
        
	<b><?php echo CHtml::encode($data->getAttributeLabel('Fecha')); ?>:</b>
	<?php echo CHtml::encode($data->Fecha); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Orden de Trabajo')); ?>:</b>
	<?php echo CHtml::encode($data->OT); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Path')); ?>:</b>
	<?php echo CHtml::encode($data->Path); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Analista')); ?>:</b>
	<?php echo CHtml::encode($data->Analista); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Tamano')); ?>:</b>
	<?php echo CHtml::encode($data->Tamano); ?>
	<br />

	<?php /*
	<b><?php echo CHtml::encode($data->getAttributeLabel('Criterio')); ?>:</b>
	<?php echo CHtml::encode($data->Criterio); ?>
	<br />

	*/ ?>

</div>
