<?php
$this->breadcrumbs = array(
	'Eventos',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	
	array('label'=>Yii::t('app', 'Gestionar') . ' Eventos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Eventos'); ?>

<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
