<div class="view">

        <b><?php echo CHtml::encode($data->getAttributeLabel('TAG')); ?>:</b>
	<?php echo CHtml::link(CHtml::encode($data->TAG), array('view', 'id'=>$data->id)); ?>
	<br />
	<b><?php echo CHtml::encode($data->getAttributeLabel('Proceso')); ?>:</b>
	<?php echo CHtml::encode($data->Proceso); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Area')); ?>:</b>
	<?php echo CHtml::encode($data->Area); ?>
	<br />

	

	<b><?php echo CHtml::encode($data->getAttributeLabel('Tablero')); ?>:</b>
	<?php echo CHtml::encode($data->Tablero); ?>
	<br />


</div>
