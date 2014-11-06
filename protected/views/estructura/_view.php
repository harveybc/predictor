<div class="view">

        <b><?php echo CHtml::encode($data->getAttributeLabel('Codigo')); ?>:</b>
<?php echo CHtml::link(CHtml::encode($data->Codigo), array('view', 'id'=>$data->id)); ?>
	<br />
        
	<b><?php echo CHtml::encode($data->getAttributeLabel('Proceso')); ?>:</b>
	       <?php echo CHtml::encode($data->Proceso); ?>
	<br />
	<b><?php echo CHtml::encode($data->getAttributeLabel('Area')); ?>:</b>
	<?php echo CHtml::encode($data->Area); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Equipo')); ?>:</b>
	<?php echo CHtml::encode($data->Equipo); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Indicativo')); ?>:</b>
	<?php echo CHtml::encode($data->Indicativo); ?>
	<br />


</div>
