<?php
$this->breadcrumbs = array(
	'Vibraciones',
	Yii::t('app', 'Index'),
);

$this->menu=array(
	array('label'=>Yii::t('app', 'Crear') . ' Registros', 'url'=>array('create')),
	array('label'=>Yii::t('app', 'Gestionar') . ' Registros', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle ('Vibraciones'); ?>
<div class="forms100c" style="text-align:left;">
<?php $this->widget('zii.widgets.CListView', array(
	'dataProvider'=>$dataProvider,
	'itemView'=>'_view',
)); ?>
</div>