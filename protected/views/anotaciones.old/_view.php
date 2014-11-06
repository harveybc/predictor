<div class="view">

	<b><?php echo CHtml::encode($data->getAttributeLabel('descripcion')); ?>:</b>
        <?php echo CHtml::link(CHtml::encode($data->descripcion), array('view', 'id'=>$data->id)); ?>
	<br />
        <b><?php echo CHtml::encode($data->getAttributeLabel('documento')); ?>:</b>
	<?php echo CHtml::encode($data->documento); ?>
	<br />
        
	<b><?php echo CHtml::encode($data->getAttributeLabel('usuario')); ?>:</b>
	<?php echo CHtml::encode($data->usuario); ?>
	<br />
     
	<b><?php echo CHtml::encode($data->getAttributeLabel('eliminado')); ?>:</b>
	<?php echo CHtml::encode($data->eliminado); ?>
	<br />


</div>
